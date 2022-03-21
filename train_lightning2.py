import sys
import os
import json 
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_v4 import ExpectedKernelModel as Model
from torch.utils.data import DataLoader, TensorDataset
from ndcg import ndcg
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

from utilities.dataset.dataloader import Scheduler
from utilities.dataset.yelp_dataset import YelpDataset as Dataset

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items

        self.true_category = self.dataset.get_reg_mat()
        config['num_group'] = int( round( self.config['num_group'] ) )

        if self.true_category.shape[0] == self.n_users:
            config['attribute'] = 'user_attribute'
            self.model = Model( self.n_users, self.n_items, self.true_category.shape[1], config['num_group'], config['num_latent']  )
        else:
            config['attribute'] = 'item_attribute'
            self.model = Model( self.n_users, self.n_items, config['num_group'], self.true_category.shape[1], config['num_latent']  )

        self.prediction_loss = nn.MarginRankingLoss( margin=config['prediction_margin'], reduction='mean' )
        self.transition_loss = nn.MarginRankingLoss( margin=config['transition_margin'], reduction='mean' )
        self.kl_div = nn.KLDivLoss( size_average='sum' )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'] )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=1 )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=1 )

    def joint_loss( self, pos_result1, neg_result1, pos_result2, neg_result2 ):
        return torch.mean( torch.relu( - ( pos_result1 - neg_result1 ) * ( pos_result2 - neg_result2 ) ) )

    def evaluate( self, true_rating, predict_rating, hr_k, recall_k, ndcg_k ):
        user_mask = torch.sum( true_rating, dim=-1 ) > 0
        predict_rating = predict_rating[ user_mask ]
        true_rating = true_rating[ user_mask ]

        _, top_k_indices = torch.topk( predict_rating, k=hr_k, dim=1, largest=True )
        hr_score = torch.mean( ( torch.sum( torch.gather( true_rating, dim=1, index=top_k_indices ), dim=-1 ) > 0 ).to( torch.float ) )

        _, top_k_indices = torch.topk( predict_rating, k=recall_k, dim=1, largest=True )

        recall_score = torch.mean( 
            torch.sum( torch.gather( true_rating, dim=1, index = top_k_indices ), dim=1 ) /
            torch.minimum( torch.sum( true_rating, dim=1 ), torch.tensor( [ recall_k ] ) )
        )

        ndcg_score = torch.mean( ndcg( predict_rating, true_rating, [ ndcg_k ] ) )

        return hr_score.item(), recall_score.item(), ndcg_score.item()

    def training_step( self, batch, batch_idx ):
        pos_interact, neg_interact = batch
        batch_size = pos_interact.shape[0]

        input_idx = torch.cat( ( pos_interact, neg_interact ), dim=0 )
        unique_user, inverse_user = torch.unique( input_idx[:,0], return_inverse=True )
        unique_item, inverse_item = torch.unique( input_idx[:,1], return_inverse=True )

        mixture, transition, user_mixture, item_mixture = self.model( unique_user, unique_item )
        print( mixture.shape, transition.shape, user_mixture.shape, item_mixture.shape )
        pos_mixture, neg_mixture = torch.hsplit( mixture[ inverse_user, inverse_item ], 2 )
        pos_transition, neg_transition = torch.hsplit( transition[ inverse_user, inverse_item ], 2 )

        l1_loss = self.prediction_loss( pos_mixture.reshape( -1, 1 ), neg_mixture.reshape( -1, 1 ), torch.ones( ( batch_size, 1 ) ) )
        l2_loss = self.transition_loss( pos_transition.reshape( -1, 1 ), neg_transition.reshape( -1, 1 ), torch.ones( ( batch_size, 1 ) ) )
        l3_loss = self.joint_loss( pos_mixture, neg_mixture, pos_transition, neg_transition )

        category_loss = None
        if self.config['attribute'] == 'item_attribute':
            category_loss = self.kl_div( torch.log( self.true_category[ unique_item ] ), item_mixture )
        elif self.config['attribute'] == 'user_attribute':
            category_loss = self.kl_div( torch.log( self.true_category[ unique_user ] ), user_mixture )

        user_distance, item_distance = self.model.mutual_distance()

        user_clustering = self.model.clustering_assignment_hardening( user_mixture )
        item_clustering = self.model.clustering_assignment_hardening( item_mixture )

        prediction_loss = l1_loss + l2_loss + l3_loss
        clustering_loss = user_clustering + item_clustering
        regularization_loss = user_distance + item_distance

        loss =  prediction_loss + self.config['gamma'] * regularization_loss - self.config['beta'] * clustering_loss
        print(loss)

        return loss

    def on_validation_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        res = self.model( batch[0][:,0], torch.arange( self.n_items ), is_test=True )
        self.y_pred = torch.vstack( ( self.y_pred, res ) )

    def on_validation_epoch_end( self ):
        val_mask, true_y = self.dataset.get_val()
        self.y_pred[ ~val_mask ] = -np.inf

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, self.y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def on_test_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def test_step( self, batch, batch_idx ):
        res = self.model( batch[0][:,0], torch.arange( self.n_items ), is_test=True )
        self.y_pred = torch.vstack( ( self.y_pred, res ) )

    def on_test_epoch_end( self ):
        test_mask, true_y = self.dataset.get_test()
        self.y_pred[ ~test_mask ] = -np.inf

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, self.y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )
        torch.save( self.y_pred, f'test_yelp_{self.config["relation"]}_non_colapse_model.pt' )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def configure_optimizers( self ):
        optimizer = optim.Adam( self.parameters(), lr=self.config['lr'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        callbacks=[
            Scheduler(),
            TuneReportCheckpointCallback( {
                'hr_score' : 'hr_score',
                'recall_score' : 'recall_score',
                'ndcg_score' : 'ndcg_score'
            },
            on='validation_end',
            filename='checkpoint'
           ),
           EarlyStopping(monitor="ndcg_score", patience=10, mode="max", min_delta=1e-3)
        ],
        progress_bar_refresh_rate=0
    )

    if checkpoint_dir:
        config["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint"
        )

    model = ModelTrainer( config, dataset )

    trainer.fit( model )

def test_model( config : dict, checkpoint_dir : str, dataset ):
    model = ModelTrainer.load_from_checkpoint( config=config, checkpoint_path=os.path.join( checkpoint_dir, 'checkpoint' ), dataset=dataset )

    trainer = pl.Trainer()
    result = trainer.test( model )

    save_json = {
        'checkpoint_dir' : str( checkpoint_dir ),
        'result' : result[0]
    }

    with open('best_model_result.json','w') as f:
        json.dump( save_json, f )

def tune_population_based( relation : str ):
    ray.init( num_cpus=10,  _temp_dir='/data2/saito/ray_tmp/' )
    dataset = ray.put( Dataset( relation ) )
    config = {
        # parameter to find
        'num_latent' : 64,
        'batch_size' : 128,
        'num_group' : tune.uniform( 4, 101 ),

        # hopefully will find right parameter
        'prediction_margin' : tune.uniform( 1, 5 ),
        'transition_margin' : tune.uniform( 0.05, 0.5 ),
        'beta' : tune.uniform( 1e-6, 1 ),
        'lr' : tune.uniform( 1e-4, 1e-1 ),
        'gamma' : tune.uniform( 1e-6, 1 ),

        # fix parameter
        'relation' : relation,
        'hr_k' : 20,
        'recall_k' : 20,
        'ndcg_k' : 100
    }

    algo = BayesOptSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=12)

    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 1 },
        metric='ndcg_score',
        mode='max',
        verbose=1,
        num_samples=1,
        config=config,
        scheduler=scheduler,
        search_alg=algo,
        name=f'non_colapse_yelp_dataset_{relation}',
        keep_checkpoints_num=2,
        local_dir=f"/data2/saito/",
        checkpoint_score_attr='ndcg_score',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )
if __name__ == '__main__':
    tune_population_based( sys.argv[1] )

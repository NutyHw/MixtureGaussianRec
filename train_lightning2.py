import sys
import os
import json 
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_v4 import KldivModel as Model
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
from utilities.dataset.ml1m_dataset import Ml1mDataset as Dataset

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

        self.kl_div = nn.KLDivLoss( size_average='batchmean' )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'] )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=1 )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=1 )

    def cross_entropy_loss( self, predict_prob, true_prob ):
        return - torch.sum( true_prob * torch.log( predict_prob ), dim=-1 )

    def gibb_sampling( self, beta, dist ):
        return torch.softmax( - beta * dist, dim=-1 )

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
        user, adj = batch

        dist, transition_prob, user_embed, item_embed = self.model( user )
        embedding_prob = self.gibb_sampling( self.config['beta'], dist )

        embedding_loss = torch.mean( self.cross_entropy_loss( embedding_prob, adj ) )
        transition_loss = torch.mean( self.cross_entropy_loss( transition_prob, adj ) )
        mutual_loss = self.kl_div( torch.log( transition_prob ), embedding_prob )

        category_loss = None
        if self.config['attribute'] == 'item_attribute':
            category_loss = self.kl_div( torch.log( item_embed ), self.true_category + 1e-6 )
        elif self.config['attribute'] == 'user_attribute':
            category_loss = self.kl_div( torch.log( user_embed ), self.true_category[ user ] + 1e-6 )

        alpha = max( self.config['alpha'] * self.config['lambda'] ** self.current_epoch, self.config['min_alpha'] )
        beta = max( self.config['beta2'] * self.config['lambda'] ** self.current_epoch, self.config['min_beta2'] )

        regularization = self.model.regularization()

        loss =  alpha * embedding_loss + beta * transition_loss + mutual_loss + self.config['gamma'] * ( torch.mean( regularization ) + category_loss )

        return loss

    def on_validation_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        res, _, _, _ = self.model( batch[0][:,0], torch.arange( self.n_items ), 'user-item' )
        res = self.gibb_sampling( self.config['beta'], res )
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
        res, _, _, _ = self.model( batch[0][:,0], torch.arange( self.n_items ), 'user-item' )
        res = self.gibb_sampling( self.config['beta'], res )
        self.y_pred = torch.vstack( ( self.y_pred, res ) )

    def on_test_epoch_end( self ):
        test_mask, true_y = self.dataset.get_test()
        self.y_pred[ ~test_mask ] = -np.inf

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, self.y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )
        torch.save( self.y_pred, f'test_ml1m_{self.config["relation"]}_model.pt' )

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
        num_sanity_val_steps=0,
        limit_train_batches=1,
        callbacks=[
            TuneReportCheckpointCallback( {
                'hr_score' : 'hr_score',
                'recall_score' : 'recall_score',
                'ndcg_score' : 'ndcg_score'
            },
            on='validation_end',
            filename='checkpoint'
           ),
           EarlyStopping(monitor="ndcg_score", patience=10, mode="max", min_delta=1e-2)
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
    ray.init( num_cpus=12,  _temp_dir='/data2/saito/ray_tmp/' )
    dataset = ray.put( Dataset( relation ) )
    config = {
        # parameter to find
        'num_latent' : 64,
        'batch_size' : 32,
        'num_group' : tune.uniform( 4, 101 ),

        # hopefully will find right parameter
        'alpha' : tune.uniform( 5, 50 ),
        'min_alpha' : tune.uniform( 1e-2, 25 ),
        'beta' : tune.uniform( 1, 50 ),
        'beta2' : tune.uniform( 5, 50 ),
        'min_beta2' : tune.uniform( 1e-2, 25 ),
        'lr' : tune.uniform( 1e-4, 1e-1 ),
        'lambda' : tune.uniform( 0.6, 1 ),
        'gamma' : tune.uniform( 1e-5, 1e-1 ),

        # fix parameter
        'relation' : relation,
        'hr_k' : 1,
        'recall_k' : 10,
        'ndcg_k' : 10
    }

    algo = BayesOptSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=12)

    scheduler = ASHAScheduler(
        grace_period=5,
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
        name=f'ml1m_dataset_{relation}',
        keep_checkpoints_num=2,
        local_dir=f"/data2/saito/",
        checkpoint_score_attr='ndcg_score',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )
if __name__ == '__main__':
    tune_population_based( sys.argv[1] )

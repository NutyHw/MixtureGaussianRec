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

from utilities.dataset.dataloader import Scheduler
from utilities.dataset.ml1m_dataset import Ml1mDataset as Dataset

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items

        self.model = Model( self.n_users, self.n_items, config['num_mixture'], config['num_mixture'], config['num_latent']  )

        self.prediction_loss = nn.MarginRankingLoss( margin=config['prediction_margin'], reduction='mean' )
        self.user_sim = self.dataset.user_sim
        self.item_sim = self.dataset.item_sim

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'] )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=16 )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=16 )

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

        unique_user, inverse_user = torch.unique( pos_interact[:,0], return_inverse=True )
        unique_item, inverse_item = torch.unique( torch.hstack( ( pos_interact[:,1], neg_interact[:,1] ) ), return_inverse=True )

        pos_inverse, neg_inverse = torch.hsplit( inverse_item, 2 )

        #user_user_sim = self.model( unique_user, unique_user, 'user-user' )
        user_item_sim, kl_div = self.model( unique_user, unique_item, 'user-item' )
        #item_item_sim = self.model( unique_item, unique_item, 'item-item' )

        #user_user_triu_indices = torch.triu_indices( user_user_sim.shape[0], user_user_sim.shape[1], offset=1 )
        #item_item_triu_indices = torch.triu_indices( item_item_sim.shape[0], item_item_sim.shape[1], offset=1 )

        #user_user_sim = user_user_sim[ user_user_triu_indices[0], user_user_triu_indices[1] ].reshape( -1, 1 )
        #item_item_sim = item_item_sim[ item_item_triu_indices[0], item_item_triu_indices[1] ].reshape( -1, 1 )

        pos_user_item_sim = user_item_sim[ inverse_user, pos_inverse ].reshape( -1, 1 )
        neg_user_item_sim = user_item_sim[ inverse_user, neg_inverse ].reshape( -1, 1 )

        ranking_loss = self.prediction_loss( pos_user_item_sim, neg_user_item_sim, torch.ones( ( pos_user_item_sim.shape[0], 1 ) ) )

        #user_comb = torch.combinations( unique_user, r=2 )
        #item_comb = torch.combinations( unique_item, r=2 )

        #inner_cluster_loss = torch.mean( user_user_sim * self.user_sim[ user_comb[:,0], user_comb[:,1] ].reshape( -1, 1 ) ) + torch.mean( item_item_sim * self.item_sim[ item_comb[:,0], item_comb[:,1] ].reshape( -1, 1 ) )

        loss = ranking_loss #+ inner_cluster_loss

        return loss + self.config['gamma'] * torch.sum( kl_div )

    def on_validation_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        res, _ = self.model( batch[0][:,0], torch.arange( self.n_items ), 'user-item' )
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
        res = self.model( batch[0][:,0], torch.arange( self.n_items ), 'user-item' )
        self.y_pred = torch.vstack( ( self.y_pred, res ) )

    def on_test_epoch_end( self ):
        test_mask, true_y = self.dataset.get_test()
        self.y_pred[ ~test_mask ] = -np.inf

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, self.y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )
        torch.save( self.y_pred, 'test_ml1m_model.pt' )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def configure_optimizers( self ):
        optimizer = optim.SGD( self.parameters(), lr=self.config['lr'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        max_epochs=256, 
        num_sanity_val_steps=0,
        callbacks=[
            TuneReportCheckpointCallback( {
                'hr_score' : 'hr_score',
                'recall_score' : 'recall_score',
                'ndcg_score' : 'ndcg_score'
            },
            on='validation_end',
            filename='checkpoint'
           ),
           EarlyStopping(monitor="ndcg_score", patience=10, mode="max", min_delta=1e-4)
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
    ray.init( num_cpus=8,  _temp_dir='/data2/saito/' )
    dataset = ray.put( Dataset( 'item_genre' ) )
    config = {
        # parameter to find
        'num_latent' : 64,
        'batch_size' : tune.grid_search([ 128, 256, 512, 1024 ]),

        # hopefully will find right parameter
        'num_mixture' : tune.grid_search([ 4, 8, 12, 16, 20 ]),
        'prediction_margin' : tune.grid_search([ 1, 2, 4 ]) ,
        'lr' : 1e-2,
        'gamma' : 1e-5,

        # fix parameter
        'relation' : relation,
        'hr_k' : 1,
        'recall_k' : 10,
        'ndcg_k' : 10
    }

    scheduler = ASHAScheduler(
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 2 },
        metric='ndcg_score',
        mode='max',
        verbose=1,
        num_samples=1,
        config=config,
        scheduler=scheduler,
        name=f'ml1m_dataset_{relation}',
        keep_checkpoints_num=2,
        local_dir=f"/data2/saito/",
        checkpoint_score_attr='ndcg_score',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )
if __name__ == '__main__':
    tune_population_based( 'item_genre' )

import sys
import os
import json 
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from models.model import Encoder, GMF
from torch.utils.data import DataLoader, TensorDataset
from ndcg import ndcg
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

from utilities.dataset.dataloader import Scheduler
from utilities.dataset.yelp_dataset import YelpDataset as Dataset

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )
        self.val_interact, self.val_test_y = self.dataset.val_interact()
        self.test_interact, _ = self.dataset.test_interact()

        user_layer = [ self.dataset.user_input.shape[1], 200, self.config['num_latent'] ]
        item_layer = [ self.dataset.item_input.shape[1], 200, self.config['num_latent'] ]

        self.user_encoder = Encoder( user_layer )
        self.item_encoder = Encoder( item_layer )
        self.gmf = GMF( self.config['num_latent'] )
        self.loss = nn.BCEWithLogitsLoss( reduction='mean' )

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

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], shuffle=True )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( self.val_interact ), batch_size=self.config['batch_size'], shuffle=False )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( self.test_interact ), batch_size=self.config['batch_size'], shuffle=False )

    def training_step( self, batch, batch_idx ):
        user_input, item_input, true_y = batch
        user_embed = self.user_encoder( user_input )
        item_embed = self.item_encoder( item_input )
        y_pred = self.gmf( user_embed * item_embed )

        return self.loss( y_pred, true_y )

    def on_validation_start( self ):
        self.y_pred = torch.zeros( ( 0, 1 ) )

    def validation_step( self, batch, batch_idx ):
        user_input, item_input = self.dataset.user_input[ batch[0][:,0] ], self.dataset.item_input[ batch[0][:,1] ]
        user_embed = self.user_encoder( user_input.to( self.device ) )
        item_embed = self.item_encoder( item_input.to( self.device ) )
        y_pred = self.gmf( user_embed * item_embed )

        self.y_pred = torch.vstack( ( self.y_pred, y_pred.cpu() ) )

    def on_validation_epoch_end( self ):
        hr_score, recall_score, ndcg_score = self.evaluate( self.val_test_y, self.y_pred.reshape( -1, 101 ), self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k']  )
        self.log_dict( {
            'hr' : hr_score,
            'recall' : recall_score,
            'ndcg' : ndcg_score
        } )

    def on_test_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, 1 ) )

    def test_step( self, batch, batch_idx ):
        user_input, item_input = self.dataset.user_input[ batch[0][:,0] ], self.dataset.item_input[ batch[0][:,1] ]
        user_embed = self.user_encoder( user_input.to( self.device ) )
        item_embed = self.item_encoder( item_input.to( self.device ) )
        y_pred = self.gmf( user_embed * item_embed )

        self.y_pred = torch.vstack( ( self.y_pred, y_pred.cpu() ) )

    def on_test_epoch_end( self ):
        hr_score, recall_score, ndcg_score = self.evaluate( self.val_test_y, self.y_pred.reshape( -1, 101 ), self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k']  )
        self.log_dict( {
            'hr' : hr_score,
            'recall' : recall_score,
            'ndcg' : ndcg_score
        } )


    def configure_optimizers( self ):
        optimizer = optim.Adam( self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'] )
        scheduler = { "scheduler" : optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='max', patience=5, threshold=1e-3 ), "monitor" : "ndcg" }
        return [ optimizer ], [ scheduler ]

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=64,
        num_sanity_val_steps=0,
        callbacks=[
            Scheduler(),
            TuneReportCheckpointCallback( {
                'hr' : 'hr',
                'recall' : 'recall',
                'ndcg' : 'ndcg'
            },
            on='validation_end',
            filename='checkpoint'
           ),
           EarlyStopping(monitor="ndcg", patience=10, mode="min", min_delta=1e-4)
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

def tune_population_based():
    ray.init( num_cpus=8, num_gpus=8 )
    dataset = ray.put( Dataset( './yelp_dataset/', 0 ) )
    config = {
        # parameter to find
        #'num_latent' : 16,
        #'weight_decay' : 1e-2,
        #'batch_size' : 32,
        'num_latent' : 64,
        'weight_decay' :tune.grid_search([ 1e-3, 1e-2, 1e-1 ]), 
        'batch_size' : tune.grid_search([ 32, 64, 128, 256 ]),
        'lr' : 1e-2,

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
        resources_per_trial={ 'cpu' : 1, 'gpu' : 1 },
        metric='ndcg',
        mode='max',
        verbose=1,
        num_samples=1,
        config=config,
        scheduler=scheduler,
        name=f'my_model_yelp_0',
        keep_checkpoints_num=2,
        local_dir=f"./",
        checkpoint_score_attr='ndcg',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )

if __name__ == '__main__':
    tune_population_based()

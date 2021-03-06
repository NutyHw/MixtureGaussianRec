import os
import json 
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vae import loss_function, MultiVAE
from utilities.dataset.dataloader import Scheduler
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
from utilities.dataset.bpr_dataset import YelpDataset as Dataset

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.dataset = ray.get( dataset )
        self.config = config

        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items
        self.val_interact, self.val_test_y = self.dataset.val_interact()
        self.test_interact, _ = self.dataset.test_interact()
        self.train_adj_mat = ( self.dataset.dataset[ 'train_adj' ] > 0 ).to( torch.float )

        if config['hidden_layer'] == 1:
            self.model = MultiVAE([ 200, 600, self.n_items ])
        elif config['hidden_layer'] == 0:
            self.model = MultiVAE([ 200, self.n_items ])

        self.normalize()

    def normalize( self ):
        row_norm = 1 / torch.sqrt( torch.sum( self.train_adj_mat, dim=1 ) ).reshape( -1, 1 )
        self.train_adj_mat = self.train_adj_mat * row_norm

    def train_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=self.config['batch_size'] )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=self.config['batch_size'], shuffle=False )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=self.config['batch_size'], shuffle=False )

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

    def on_train_epoch_start( self ):
        self.model.train()

    def training_step( self, batch, batch_idx ):
        batch_data = self.train_adj_mat[ batch[0].squeeze() ].to( self.device )
        recon_batch, mu, logvar = self.model(batch_data)
        loss = loss_function( recon_batch, batch_data, mu, logvar, self.config['anneal'] )

        self.log_dict({ 'loss' : loss.item() })

        return loss

    def on_validation_epoch_start( self ):
        self.model.eval()
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        batch_data = self.train_adj_mat[ batch[0].squeeze() ].to( self.device )
        recon_batch, mu, logvar = self.model(batch_data)
        self.y_pred = torch.vstack( ( self.y_pred, recon_batch.cpu() ) )

    def on_validation_epoch_end( self ):
        y_pred = torch.gather( self.y_pred, 1, self.val_interact )

        hr_score, recall_score, ndcg_score = self.evaluate( self.val_test_y, y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def on_test_epoch_start( self ):
        self.model.eval()
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def test_step( self, batch, batch_idx ):
        batch_data = self.train_adj_mat[ batch[0].squeeze() ]
        recon_batch, mu, logvar = self.model(batch_data)
        self.y_pred = torch.vstack( ( self.y_pred, recon_batch.cpu() ) )

    def on_test_epoch_end( self ):
        y_pred = torch.gather( self.y_pred, 1, self.test_interact )

        hr_score, recall_score, ndcg_score = self.evaluate( self.val_test_y, y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def configure_optimizers( self ):
        optimizer = optim.Adam( self.parameters(), lr=self.config['lr'], weight_decay=self.config['gamma'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=64,
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
        ]
    )

    model = ModelTrainer( config, dataset=dataset )

    trainer.fit( model )


def test_model( config : dict, checkpoint_dir : str, dataset ):
    model = ModelTrainer.load_from_checkpoint( config=config, checkpoint_path=os.path.join( checkpoint_dir, 'checkpoint' ), dataset=dataset )

    trainer = pl.Trainer()
    result = trainer.test( model )

    save_json = {
        'checkpoint_dir' : str( checkpoint_dir ),
        'result' : result[0]
    }

    with open('best_model_vae.json','w') as f:
        json.dump( save_json, f )

def tune_model():
    ray.init( num_cpus=8, num_gpus=8 )
    dataset = ray.put( Dataset( './yelp_dataset/', '1' ) )
    config = {
        # grid search parameter
        'batch_size' : tune.grid_search([ 128, 256, 512, 1024 ]),
        'lr' : tune.grid_search([ 1e-4, 1e-3, 1e-2, 1e-1 ]),

        # hopefully will find right parameter
        'hidden_layer' : tune.grid_search([ 0, 1 ]),
        'anneal' : tune.uniform( 0.1, 0.6 ),
        'gamma' : tune.grid_search([ 1e-4, 1e-3, 1e-2, 1e-1 ]),

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
        metric='ndcg_score',
        mode='max',
        num_samples=1,
        verbose=1,
        config=config,
        scheduler=scheduler,
        name=f'yelp_vae_1',
        local_dir=".",
        keep_checkpoints_num=1, 
        checkpoint_score_attr='ndcg_score'
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset=dataset )

if __name__ == '__main__':
    tune_model()

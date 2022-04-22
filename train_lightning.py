import sys
import os
import json 
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from models.model import Encoder, Decoder
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
        layer = [ self.dataset.attribute.shape[1], self.config['num_latent'] ]

        self.encoder = Encoder( layer )
        self.decoder = Decoder( layer[::-1] )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], shuffle=True )

    def training_step( self, batch, batch_idx ):
        input = batch
        embed = self.encoder( input )
        pred = self.decoder( embed )

        loss = torch.norm( embed - pred )
        return loss

    def on_validation_start( self, pred ):
        self.loss = 0
        self.counter = 0

    def validation_step( self, batch, batch_idx ):
        input = batch
        embed = self.encoder( input )
        pred = self.decoder( embed )

        self.loss += torch.norm( embed - pred )
        self.counter += 1

    def on_validation_end( self ):
        self.log_dict({ 'loss' : self.loss / self.counter })

    def configure_optimizers( self ):
        optimizer = optim.Adam( self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=128,
        callbacks=[
            TuneReportCheckpointCallback( {
                'hr_score' : 'hr_score',
                'recall_score' : 'recall_score',
                'ndcg_score' : 'ndcg_score'
            },
            on='validation_end',
            filename='checkpoint'
           ),
           EarlyStopping(monitor="loss", patience=5, mode="min", min_delta=1e-2)
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
    dataset = YelpDataset( './yelp_dataset/', 0, 'BCat', 40 )
    config = {
        # parameter to find
        'num_latent' : tune.grid_search([ 16, 32, 64, 128 ]),
        'batch_size' : 32,
        'weight_decay' :tune.grid_search([ 1e-3, 1e-2, 1e-1 ]), 
        'lambda' : tune.grid_search([ 1e-3, 1e-2, 1e-1 ]),
        'lr' : 1e-2,

        # fix parameter
        'hr_k' : 20,
        'recall_k' : 20,
        'ndcg_k' : 100
    }

    scheduler = ASHAScheduler(
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 1, 'gpu' : 1 },
        metric='loss',
        mode='min',
        verbose=1,
        num_samples=1,
        config=config,
        scheduler=scheduler,
        name=f'pretrain_yelp',
        keep_checkpoints_num=2,
        local_dir=f"./",
        checkpoint_score_attr='loss',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )
if __name__ == '__main__':
    tune_population_based()

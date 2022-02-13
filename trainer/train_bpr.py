import os
import json 
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bpr import BPR
from utilities.dataset.dataloader import Scheduler
from utilities.dataset.ml1m_dataset import Ml1mDataset
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

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )

        self.model = BPR( self.dataset.n_users, self.dataset.n_items, config['num_latent'] )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], num_workers=2 )

    def val_dataloader( self ):
        x = self.dataset.get_val()
        y = torch.zeros( ( x.shape[0] // 101, 101 ) )
        y[ :, 0 ] = 1
        y = y.reshape( -1, 1 )

        return DataLoader( TensorDataset( x, y ), batch_size=self.config['batch_size'], shuffle=False, num_workers=2 )

    def test_dataloader( self ):
        x = self.dataset.get_test()
        y = torch.zeros( ( x.shape[0] // 101, 101 ) )
        y[ :, 0 ] = 1
        y = y.reshape( -1, 1 )

        return DataLoader( TensorDataset( x, y ), batch_size=self.config['batch_size'], shuffle=False, num_workers=2 )

    def evaluate( self, true_rating, predict_rating ):
        _, top_k_indices = torch.topk( predict_rating, k=1, dim=1, largest=True )
        hr_1 = torch.mean( torch.gather( true_rating, dim=-1, index=top_k_indices ) )

        _, top_k_indices = torch.topk( predict_rating, k=10, dim=-1, largest=True )

        recall_10 = torch.mean( 
            torch.sum( torch.gather( true_rating, dim=1, index = top_k_indices ), dim=-1 )
        )

        ndcg_10 = torch.mean( ndcg( predict_rating, true_rating, [ 10 ] ) )

        return hr_1.item(), recall_10.item(), ndcg_10.item()

    def training_step( self, batch, batch_idx ):
        pos_interact, neg_interact = batch
        batch_size = pos_interact.shape[0]

        input_idx = torch.cat( ( pos_interact, neg_interact ), dim=0 )
        res = self.model( input_idx[:,0], input_idx[:,1] )
        pos_prob, neg_prob = torch.split( res, split_size_or_sections=batch_size, dim=0 )

        # prediction loss
        loss = F.logsigmoid( pos_prob - neg_prob ).sum()

        # regularization loss
        user_embed = self.model.W[ input_idx[:,0] ]
        item_embed = self.model.H[ input_idx[:,1] ]

        reg_loss = self.config['weight_decay'] * ( 
            user_embed.norm(dim=1).pow(2).sum() +
            item_embed.norm(dim=1).pow(2).sum()
        )

        return - loss + reg_loss

    def on_validation_epoch_start( self ):
        self.predict_score = torch.zeros( ( 0, 1 ) )
        self.true_score = torch.zeros( ( 0, 1 ) )

    def validation_step( self, batch, batch_idx ):
        interact, y = batch
        res = self.model( interact[:,0], interact[:,1] )
        self.predict_score = torch.vstack( ( self.predict_score, res ) )
        self.true_score = torch.vstack( ( self.true_score, y ) )

    def on_validation_epoch_end( self ):
        self.true_score = self.true_score.reshape( -1, 101 )
        self.predict_score = self.predict_score.reshape( -1, 101 )
        hr_1, recall_10, ndcg_10 = self.evaluate( self.true_score, self.predict_score )
        self.log_dict({
            'hr_1' : hr_1,
            'recall_10' : recall_10,
            'ndcg_10' : ndcg_10
        })

    def on_test_epoch_start( self ):
        self.predict_score = torch.zeros( ( 0, 1 ) )
        self.true_score = torch.zeros( ( 0, 1 ) )

    def test_step( self, batch, batch_idx ):
        interact, y = batch
        res = self.model( interact[:,0], interact[:,1] )
        self.predict_score = torch.vstack( ( self.predict_score, res ) )
        self.true_score = torch.vstack( ( self.true_score, y ) )

    def on_test_epoch_end( self ):
        self.true_score = self.true_score.reshape( -1, 101 )
        self.predict_score = self.predict_score.reshape( -1, 101 )
        hr_1, recall_10, ndcg_10 = self.evaluate( self.true_score, self.predict_score )
        self.log_dict({
            'hr_1' : hr_1,
            'recall_10' : recall_10,
            'ndcg_10' : ndcg_10
        })

    def configure_optimizers( self ):
        optimizer = optim.Adam( self.parameters(), lr=self.config['lr'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        max_epochs=128, 
        num_sanity_val_steps=0,
        callbacks=[
            Scheduler(),
            TuneReportCheckpointCallback( {
                'hr_1' : 'hr_1',
                'recall_10' : 'recall_10',
                'ndcg_10' : 'ndcg_10'
            },
            on='validation_end',
            filename='checkpoint'
           ),
           EarlyStopping(monitor="ndcg_10", patience=10, mode="max", min_delta=0.01)
        ],
        progress_bar_refresh_rate=0
    )

    model = ModelTrainer( config, dataset )

    trainer.fit( model )

def test_model( config : dict, checkpoint_dir : str, dataset ):
    print( checkpoint_dir )
    model = ModelTrainer.load_from_checkpoint( config=config, checkpoint_path=os.path.join( checkpoint_dir, 'checkpoint' ), dataset=dataset )

    trainer = pl.Trainer()
    result = trainer.test( model )

    save_json = {
        'checkpoint_dir' : checkpoint_dir,
        'result' : result[0]
    }

    with open('best_model_result.json','w') as f:
        json.dump( save_json, f )

def tune_model( relation_id : int ):
    ray.init( num_cpus=2 )
    dataset = ray.put( Ml1mDataset( relation_id ) )
    config = {
        # grid search parameter
        'num_latent' : tune.choice([ 8, 16, 32 ]),
        'weight_decay' : tune.choice([ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ]),

        # hopefully will find right parameter
        'batch_size' : tune.choice([ 128, 256, 512, 1024 ]),
        'lr' : tune.quniform( 1e-3, 1e-2, 1e-3 )
    }

    scheduler = ASHAScheduler(
        max_t=1,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter( 
        parameter_columns=[ 'num_latent', 'weight_decay' ],
        metric_columns=[ 'hr_1', 'recall_10', 'ndcg_10'  ]
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 1 },
        metric='ndcg_10',
        mode='max',
        num_samples=2,
        verbose=1,
        config=config,
        progress_reporter=reporter,
        scheduler=scheduler,
        name=f'ml1m_bpr',
        local_dir=".",
        keep_checkpoints_num=1, 
        checkpoint_score_attr='ndcg_10'
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset=dataset )

if __name__ == '__main__':
    tune_model( 0 )

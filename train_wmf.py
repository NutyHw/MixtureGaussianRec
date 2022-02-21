import os
import json 
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wmf import WMF  
from utilities.dataset.dataloader import Scheduler
from utilities.dataset.non_pair_yelp_dataset import YelpDataset
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
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items

        self.model = WMF( self.n_users, self.n_items, config['num_latent'] )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], num_workers=16 )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( 1 ).reshape( -1, 1 ) ), batch_size=1, shuffle=False, num_workers=16 )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( 1 ).reshape( -1, 1 ) ), batch_size=1, shuffle=False, num_workers=16 )

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
        interact, y = batch
        res = self.model( interact[:,0], interact[:,1] )
        loss = torch.sum( ( self.config['C'] + 1 ) * y * ( y - res ) ** 2 )

        self.log_dict({ 'loss' : loss.item() })

        return loss

    def validation_step( self, batch, batch_idx ):
        val_mask, true_y = self.dataset.get_val()
        y_pred = self.model( None, None, is_test=True )
        y_pred[ ~val_mask ] = 0

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def test_step( self, batch, batch_idx ):
        test_mask, true_y = self.dataset.get_test()
        y_pred = self.model( None, None, is_test=True )
        y_pred[ ~test_mask ] = 0

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def configure_optimizers( self ):
        optimizer = optim.SGD( self.parameters(), lr=self.config['lr'], weight_decay=self.config['gamma'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        max_epochs=256, 
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
           EarlyStopping(monitor="ndcg_score", patience=10, mode="max", min_delta=1e-4)
        ],
        progress_bar_refresh_rate=0
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

def tune_model():
    ray.init( num_cpus=10 )
    dataset = YelpDataset()
    dataset = ray.put( dataset )
    config = {
        # grid search parameter
        'num_latent' : 16,
        'gamma' : tune.uniform(1e-5,1e-1),

        # hopefully will find right parameter
        'batch_size' : tune.choice([ 128, 256, 512, 1024 ]),
        'C' : tune.randint(5,100),
        'lr' : tune.quniform( 1e-3, 1e-2, 1e-3 ),

        'hr_k' : 20,
        'recall_k' : 20,
        'ndcg_k' : 100
    }

    scheduler = ASHAScheduler(
        max_t=256,
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 2 },
        metric='ndcg_score',
        mode='max',
        num_samples=2,
        verbose=1,
        config=config,
        scheduler=scheduler,
        name=f'yelp_wmf',
        local_dir=".",
        keep_checkpoints_num=1, 
        checkpoint_score_attr='ndcg_score'
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )

if __name__ == '__main__':
    tune_model()

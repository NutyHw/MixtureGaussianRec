import os
import json 
from tqdm import tqdm
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.bpr import BPR
from utilities.dataset.bpr_dataloader import Scheduler
from utilities.dataset.bpr_dataset import YelpDataset as Dataset
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

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.dataset = ray.get( dataset )
        self.val_interact, self.val_test_y = self.dataset.val_interact()
        self.test_interact, _ = self.dataset.test_interact()
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items
        self.config = config

        self.model = BPR( self.n_users, self.n_items, config['num_latent'] )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'] )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( 1 ).reshape( -1, 1 ) ), batch_size=1 )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( 1 ).reshape( -1, 1 ) ), batch_size=1 )

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
        res = self.model( input_idx[:,0], input_idx[:,1] )
        pos_res_out, neg_res_out = torch.split( res, split_size_or_sections=batch_size, dim=0 )

        loss = - torch.sum( torch.log( torch.sigmoid( pos_res_out - neg_res_out ) ) )

        #self.log_dict({ 'loss' : loss.item() })

        return loss

    def validation_step( self, batch, batch_idx ):
        pass

    def on_validation_epoch_end( self ):
        y_pred = self.model( None, None, is_test=True ).cpu()
        y_pred = torch.gather( y_pred, 1, self.val_interact )

        hr_score, recall_score, ndcg_score = self.evaluate( self.val_test_y, y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        metric = {
            'hr' : hr_score,
            'recall' : recall_score,
            'ndcg' : ndcg_score
        }

        self.log_dict({
            'hr' : hr_score,
            'recall' : recall_score,
            'ndcg' : ndcg_score
        })

        return metric


    def test_step( self, batch, batch_idx ):
        y_pred = self.model( None, None, is_test=True ).cpu()
        y_pred = torch.gather( y_pred, 1, self.test_interact )

        hr_score, recall_score, ndcg_score = self.evaluate( self.val_test_y, y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr' : hr_score,
            'recall' : recall_score,
            'ndcg' : ndcg_score
        })

    def configure_optimizers( self ):
        optimizer = optim.SGD( self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'] )
        return optimizer

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
           EarlyStopping(monitor="ndcg", patience=10, mode="max", min_delta=1e-4)
        ]
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
    ray.init( num_cpus=8, num_gpus=8 )
    dataset = ray.put( Dataset( './yelp_dataset/', '1' ) )
    config = {
        # grid search parameter
        'num_latent' : 64,
        'weight_decay' : tune.grid_search([ 1e-4, 1e-3, 1e-2, 1e-1 ]),
        'batch_size' : tune.grid_search([ 128, 256, 512, 1024 ]),
        'lr' : tune.grid_search([ 1e-4, 1e-3, 1e-2, 1e-1 ]),

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
        num_samples=1,
        verbose=1,
        config=config,
        scheduler=scheduler,
        name=f'yelp_bpr_1',
        local_dir=".",
        keep_checkpoints_num=1, 
        checkpoint_score_attr='ndcg'
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )

if __name__ == '__main__':
    tune_model()

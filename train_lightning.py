import os
import sys
import json 
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torch.jit as jit
from torch.utils.data import DataLoader
from models.model import Model

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

if sys.argv[1] == 'ml1m':
    from utilities.dataset.ml1m_dataset import Ml1mDataset as Dataset
elif sys.argv[1] == 'yelp':
    from utilities.dataset.yelp_dataset import YelpDataset as Dataset

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items
        self.reg_mat = self.dataset.get_reg_mat()

        config['num_user'] = self.n_users
        config['num_item'] = self.n_items
        config['num_category'] = self.reg_mat.shape[1]
        config['num_group'] = int( round( self.config['num_group'] ) )
        if self.reg_mat.shape[0] == self.dataset.n_users:
            config['attribute'] = 'user_attribute'
        elif self.reg_mat.shape[0] == self.dataset.n_items:
            config['attribute'] = 'item_attribute'

        model = Model( **config )
        self.model = jit.trace( model, torch.tensor([ 0, 1 ]) )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'] )

    def val_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], shuffle=False )

    def test_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], shuffle=False )

    def negative_log_likehood( self, energy_res, x_adj, beta ):
        return torch.sum( energy_res * x_adj, dim=-1 ) \
            + ( 1 / beta ) * torch.log( torch.sum( torch.exp( - beta * energy_res ), dim=-1 ) )

    def compute_joint_distillation( self, energy_res, transition_prob, beta ):
        return F.kl_div( 
            torch.log( torch.softmax( - energy_res * beta, dim=-1 ) ),
            transition_prob,
            reduction='sum'
        )

    def compute_cross_entropy( self, predict_prob, true_prob ):
        return - torch.sum( true_prob * torch.log( predict_prob ), dim=1 )

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
        user_idx, x_adj, prob_adj = batch

        norm_true_prob = x_adj / torch.sum( x_adj, dim=-1 ).reshape( -1, 1 )

        mixture_prob, transition_prob, regularization, user_embedding, item_embedding = self.model( user_idx )

        alpha = max( self.config['alpha_beta'] * self.config['lambda'] ** self.current_epoch, self.config['min_alpha_beta'] )

        # compute loss
        l1 = torch.sum( self.compute_cross_entropy( mixture_prob, norm_true_prob ) )
        l2 = torch.sum( self.compute_cross_entropy( transition_prob, norm_true_prob ) )
        l3 = F.kl_div( torch.log( transition_prob ), mixture_prob, reduction='sum' )

        loss = l1 * alpha + l2 * alpha + l3

        # regularization
        norm_regularization = torch.sum( regularization )
        attribute_regularization = 0

        if self.config['attribute'] == 'item_attribute':
            attribute_regularization = F.kl_div( torch.log( self.reg_mat + 1e-6 ), item_embedding, reduction='sum' )
        elif self.config['attribute'] == 'user_attribute':
            attribute_regularization = F.kl_div( torch.log( self.reg_mat[ user_idx ] + 1e-6 ), user_embedding, reduction='sum' )

        reg_loss = self.config['gamma'] * ( norm_regularization + attribute_regularization )

        return loss + reg_loss

    def on_validation_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        user_idx, _, _ = batch
        mixture_prob, transition_prob, regularization, user_embedding, item_embedding = self.model( user_idx )

        self.y_pred = torch.vstack( ( self.y_pred, mixture_prob ) )

    def on_validation_epoch_end( self ):
        val_mask, true_y = self.dataset.get_val()
        self.y_pred[ ~val_mask ] = 0

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, self.y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

        self.y_pred = None

    def on_test_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def test_step( self, batch, batch_idx ):
        user_idx, x_adj, prob_adj = batch
        mixture_prob, transition_prob, regularization, user_embedding, item_embedding = self.model( user_idx )

        self.y_pred = torch.vstack( ( self.y_pred, mixture_prob ) )

    def on_test_epoch_end( self ):
        test_mask, true_y = self.dataset.get_test()
        self.y_pred[ ~test_mask ] = 0
        torch.save( self.y_pred, f'{self.config["relation"]}_predict.pt' )

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, self.y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

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
           EarlyStopping(monitor="ndcg_score", patience=5, mode="max", min_delta=1e-2)
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
    ray.init( num_cpus=10, _temp_dir='/data2/saito/ray_tmp/' )
    dataset = ray.put( Dataset( relation ) )
    config = {
        # parameter to find
        'num_latent' : 64,
        'batch_size' : 32,

        # hopefully will find right parameter
        'num_group' : tune.uniform(4,100),
        'gamma' : tune.uniform( 1e-5, 1e-1 ),
        'lr' : tune.uniform( 1e-4, 1e-1 ),
        'alpha_beta' : 1,
        'min_alpha_beta' : 0.01,
        'beta' : tune.uniform( 1e-2, 5 ),
        'lambda' : tune.uniform( 0.8, 1 ),

        # fix parameter
        'relation' : relation,
        'hr_k' : 1,
        'recall_k' : 10,
        'ndcg_k' : 10
    }

    algo = BayesOptSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=10)

    scheduler = ASHAScheduler(
        grace_period=5,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 1 },
        metric='ndcg_score',
        mode='max',
        num_samples=100,
        verbose=1,
        config=config,
        scheduler=scheduler,
        search_alg=algo,
        name=f'{sys.argv[1]}_dataset_{relation}',
        keep_checkpoints_num=2,
        local_dir=f"/data2/saito/",
        checkpoint_score_attr='ndcg_score',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )

if __name__ == '__main__':
    tune_population_based( sys.argv[2] )

import os
import json 
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model
from models.ensemble_model import EnsembleModel
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
from utilities.dataset.ml1m_dataset import Ml1mDataset as Dataset

class EnsembleTrainer( pl.LightningModule ):
    def __init__( self, dataset, based_model, config ):
        super().__init__()
        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items
        self.config = config
        self.dataset = ray.get( dataset )
        self.model = EnsembleModel( num_user=self.n_users, num_item=self.n_items, based_model=ray.get( based_model ) )

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'] )

    def val_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], shuffle=False )

    def test_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], shuffle=False )

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

        mixture_prob, transition_prob = self.model( user_idx )

        alpha = max( self.config['alpha'] * self.config['lambda'] ** self.current_epoch, self.config['min_alpha'] )
        beta = max( self.config['beta2'] * self.config['lambda'] ** self.current_epoch, self.config['min_beta2'] )

        # compute loss
        l1 = torch.sum( self.compute_cross_entropy( mixture_prob, norm_true_prob ) )
        l2 = torch.sum( self.compute_cross_entropy( transition_prob, norm_true_prob ) )
        l3 = F.kl_div( transition_prob, mixture_prob, reduction='sum' )

        loss = alpha * l1 + beta * l2 + l3

        return loss

    def on_validation_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        user_idx, _, _ = batch
        mixture_prob, _ = self.model( user_idx )

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

    def on_test_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def test_step( self, batch, batch_idx ):
        user_idx, _, _ = batch
        mixture_prob, _ = self.model( user_idx )

        self.y_pred = torch.vstack( ( self.y_pred, mixture_prob ) )

    def on_test_epoch_end( self ):
        test_mask, true_y = self.dataset.get_test()
        self.y_pred[ ~test_mask ] = 0

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, self.y_pred, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score
        })

    def configure_optimizers( self ):
        optimizer = optim.Adam( self.parameters(), lr=self.config['lr'] )
        return optimizer

def load_models( checkpoint_dir ):
    params = None

    with open( os.path.join( checkpoint_dir, 'params.json' ) ) as f:
        params = json.load( f )

    state_dict = torch.load( os.path.join( checkpoint_dir, 'checkpoint' ) )['state_dict']
    new_state_dict = dict()

    for key in state_dict.keys():
        new_state_dict[ key[6:] ] = state_dict[ key ]

    params['num_user'] = new_state_dict['embedding.user_embedding'].shape[0]
    params['num_item'] = new_state_dict['embedding.item_embedding'].shape[0]
    params['num_group'] = new_state_dict['embedding.group_mu'].shape[0]
    params['num_category'] = new_state_dict['embedding.category_mu'].shape[0]

    model = Model( **params )
    model.load_state_dict( state_dict=new_state_dict )

    return model

def test_model( config : dict, checkpoint_dir : str, dataset, based_model ):
    model = EnsembleTrainer.load_from_checkpoint( config=config, checkpoint_path=os.path.join( checkpoint_dir, 'checkpoint' ), dataset=dataset, based_model=based_model)

    trainer = pl.Trainer()
    result = trainer.test( model )

    save_json = {
        'checkpoint_dir' : str( checkpoint_dir ),
        'result' : result[0]
    }

    with open('best_model_result.json','w') as f:
        json.dump( save_json, f )

def train_model( config, checkpoint_dir=None, dataset=None, based_model=None):
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
           EarlyStopping(monitor="ndcg_score", patience=10, mode="max", min_delta=1e-3)
        ],
        progress_bar_refresh_rate=0
    )

    model = EnsembleTrainer( dataset=dataset, based_model=based_model, config=config )
    trainer.fit( model )

def tune_model( best_model_path_1 : str, best_model_path_2 : str ):
    ray.init( num_cpus=1 )
    dataset = ray.put( Dataset( 'item_genre' ) )
    based_model = ray.put( [ load_models( best_model_path_1 ), load_models (best_model_path_2)  ] )

    config = {
        # parameter to find
        'batch_size' : 128,

        # hopefully will find right parameter
        'lr' : tune.uniform( 1e-4, 1e-1 ),
        'alpha' : tune.uniform( 5, 50 ),
        'min_alpha' : tune.uniform( 1e-2, 25 ),
        'beta2' : tune.uniform( 5, 50 ),
        'min_beta2' : tune.uniform( 1e-2, 25 ),
        'lambda' : tune.uniform( 0.6, 1 ),

        # fix parameter
        'hr_k' : 1,
        'recall_k' : 10,
        'ndcg_k' : 10
    }

    scheduler = ASHAScheduler(
        max_t=256,
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset, based_model=based_model),
        resources_per_trial={ 'cpu' : 1 },
        metric='ndcg_score',
        mode='max',
        num_samples=1,
        verbose=1,
        config=config,
        scheduler=scheduler,
        name=f'ml1m_ensemble_model',
        local_dir=".",
        keep_checkpoints_num=1, 
        checkpoint_score_attr='ndcg_score'
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset, model1, model2 )

if __name__ == '__main__':
    tune_model( './best_models/yelp/BCat/', './best_models/yelp/BCity/' )

import os
import json 
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model
from models.ensemble_model import EnsembleModel
from utilities.dataset.dataloader import Scheduler
from utilities.dataset.yelp_dataset import YelpDataset
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

class EnsembleTrainer( pl.LightningModule ):
    def __init__( self, dataset, model1, model2, config ):
        super().__init__()
        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items
        self.config = config
        self.dataset = ray.get( dataset )
        self.ensemble_model = EnsembleModel( self.dataset.n_users, ray.get( model1 ), ray.get( model2 ) )

        self.prediction_loss = nn.MarginRankingLoss( margin=config['prediction_margin'], reduction='sum' )
        self.transition_loss = nn.MarginRankingLoss( margin=config['transition_margin'], reduction='sum' )

        self.alpha = config['alpha']
        self.beta = config['beta']

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], num_workers=2 )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=16 )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False, num_workers=16 )

    def joint_loss( self, pos_result1, neg_result1, pos_result2, neg_result2 ):
        return torch.mean( torch.relu( - ( pos_result1 - neg_result1 + self.config['prediction_margin'] ) * ( pos_result2 - neg_result2 + self.config['transition_margin'] ) ) )

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
        res = self.ensemble_model( input_idx[:,0], input_idx[:,1] )

        pos_res_out, neg_res_out = torch.split( res['out'], split_size_or_sections=batch_size, dim=0 )
        pos_res_kg_prob, neg_res_kg_prob = torch.split( res['kg_prob'], split_size_or_sections=batch_size, dim=0 )

        # prediction loss
        l1_loss = self.prediction_loss( pos_res_out, neg_res_out, torch.ones( ( batch_size, 1 ) ) )
        l2_loss = self.transition_loss( pos_res_kg_prob, neg_res_kg_prob, torch.ones( ( batch_size, 1 ) ) )
        l3_loss = self.joint_loss( pos_res_out, neg_res_out, pos_res_kg_prob, neg_res_kg_prob )

        # regularization loss
        loss = l1_loss * self.alpha + l2_loss * self.beta + l3_loss
        self.log_dict({ 'loss' : loss.item() })

        return loss

    def on_validation_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        res = self.ensemble_model( batch[0][:,0], None, is_test=True )

        self.y_pred = torch.vstack( ( self.y_pred, res ) )

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
        res = self.ensemble_model( batch[0][:,0], None, is_test=True )

        self.y_pred = torch.vstack( ( self.y_pred, res ) )

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

    print( params['num_user'], params['num_item'] )
    model = Model( **params )
    model.load_state_dict( state_dict=new_state_dict )

    return model

def test_model( config : dict, checkpoint_dir : str, dataset, model1, model2 ):
    model = EnsembleTrainer.load_from_checkpoint( config=config, checkpoint_path=os.path.join( checkpoint_dir, 'checkpoint' ), dataset=dataset, model1=model1, model2=model2 )

    trainer = pl.Trainer()
    result = trainer.test( model )

    save_json = {
        'checkpoint_dir' : str( checkpoint_dir ),
        'result' : result[0]
    }

    with open('best_model_result.json','w') as f:
        json.dump( save_json, f )

def train_model( config, checkpoint_dir=None, dataset=None, model1=None, model2=None ):
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

    model = EnsembleTrainer( dataset, model1, model2, config )

    trainer.fit( model )

def tune_model( best_model_path_1 : str, best_model_path_2 : str ):
    ray.init( num_cpus=1 )
    dataset = ray.put( YelpDataset() )
    model1, model2 = ray.put( load_models( best_model_path_1 ) ), ray.put( load_models (best_model_path_2) )

    config = {
        # hopefully will find right parameter
        'batch_size' : 32,
        'prediction_margin' : tune.uniform( 1, 5 ),
        'transition_margin' : tune.uniform( 0.01, 0.5 ),
        'lr' : tune.quniform( 1e-3, 1e-2, 1e-3 ),
        'alpha' : tune.quniform( 10, 200, 10 ),
        'beta' : tune.qrandint( 10, 100, 10 ),
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
        partial( train_model, dataset=dataset, model1=model1, model2=model2 ),
        resources_per_trial={ 'cpu' : 1 },
        metric='ndcg_score',
        mode='max',
        num_samples=200,
        verbose=1,
        config=config,
        scheduler=scheduler,
        name=f'yelp_ensemble_model',
        local_dir="/data2/saito/",
        keep_checkpoints_num=1, 
        checkpoint_score_attr='ndcg_score'
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset, model1, model2 )

if __name__ == '__main__':
    tune_model( './best_models/yelp/BCat/', './best_models/yelp/BCity/' )

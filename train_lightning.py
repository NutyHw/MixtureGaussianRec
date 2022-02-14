import os
import json 
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model
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
from utilities.dataset.yelp_dataset import YelpDataset as Dataset

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items
        self.reg_mat = self.dataset.get_reg_mat( config['relation'] )
        self.model = Model( self.n_users, self.n_items, self.reg_mat.shape[1], config['num_group'], config['num_latent'] )

        self.prediction_loss = nn.MarginRankingLoss( margin=config['prediction_margin'], reduction='sum' )
        self.transition_loss = nn.MarginRankingLoss( margin=config['transition_margin'], reduction='sum' )

        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], num_workers=1 )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users ).reshape( -1, 1 ) ), batch_size=256, shuffle=False )

    def joint_loss( self, pos_result1, neg_result1, pos_result2, neg_result2 ):
        return torch.mean( torch.relu( - ( pos_result1 - neg_result1 + self.config['prediction_margin'] ) * ( pos_result2 - neg_result2 + self.config['transition_margin'] ) ) )

    def normal_regularization( self, dist : torch.Tensor ):
        mu, sigma = torch.hsplit( dist, 2 )
        sigma = F.elu( sigma ) + 1

        # mu regularization
        return torch.mean( torch.sqrt( torch.sum( mu ** 2, dim=-1 ) ) )

    def normal_regularization( self, dist : torch.Tensor ):
        mu, sigma = torch.hsplit( dist, 2 )

        return 0.5 * torch.sum( (
            torch.sum( mu ** 2, dim=1 ) +\
            torch.sum( sigma, dim=1 ) -\
            self.config['num_latent'] -\
            torch.log( torch.prod( sigma, dim=-1 ) )
        ) )

    def evaluate( self, true_rating, predict_rating, hr_k, recall_k, ndcg_k ):
        _, top_k_indices = torch.topk( predict_rating, k=hr_k, dim=1, largest=True )
        hr_score = torch.mean( ( torch.gather( true_rating, dim=1, index=top_k_indices ) > 0 ).to( torch.float ) )

        _, top_k_indices = torch.topk( predict_rating, k=recall_k, dim=1, largest=True )

        recall_score = torch.nanmean( 
            torch.sum( torch.gather( true_rating, dim=1, index = top_k_indices ), dim=-1 ) / 
            torch.minimum( torch.sum( true_rating, dim=1 ), torch.tensor( [ recall_k ] ) )
        )

        ndcg_score = torch.mean( ndcg( predict_rating, true_rating, [ ndcg_k ] ) )

        return hr_score.item(), recall_score.item(), ndcg_score.item()

    def training_step( self, batch, batch_idx ):
        pos_interact, neg_interact = batch
        batch_size = pos_interact.shape[0]

        input_idx = torch.cat( ( pos_interact, neg_interact ), dim=0 )
        res = self.model( input_idx[:,0], input_idx[:,1] )

        pos_res_out, neg_res_out = torch.split( res['out'], split_size_or_sections=batch_size, dim=0 )
        pos_res_kg_prob, neg_res_kg_prob = torch.split( res['kg_prob'], split_size_or_sections=batch_size, dim=0 )

        # prediction loss
        l1_loss = self.prediction_loss( pos_res_out, neg_res_out, torch.ones( ( batch_size, 1 ) ) )
        l2_loss = self.transition_loss( pos_res_kg_prob, neg_res_kg_prob, torch.ones( ( batch_size, 1 ) ) )
        l3_loss = self.joint_loss( pos_res_out, neg_res_out, pos_res_kg_prob, neg_res_kg_prob )

        # regularization loss
        item_idx = torch.unique( input_idx[:,1], sorted=True )
        category_reg = F.kl_div( res['category_kg'], self.reg_mat[ item_idx ] )
        normal_loss = self.normal_regularization( res['distribution'][0] ) + self.normal_regularization( res['distribution'][1] )

        loss = l1_loss * self.alpha + l2_loss * self.beta + l3_loss + self.gamma * ( category_reg + normal_loss )

        self.log_dict({ 'loss' : loss.item() })

        return loss

    def on_train_epoch_end( self ):
        self.alpha = max( self.alpha * self.config['lambda'], self.config['alpha'] * self.config['min_lambda'] )
        self.beta = max( self.beta * self.config['lambda'], self.config['beta'] * self.config['min_lambda'] )

    def on_validation_epoch_start( self ):
        self.y_pred = torch.zeros( ( 0, self.n_items ) )

    def validation_step( self, batch, batch_idx ):
        res = self.model( batch[0][:,0], None, is_test=True )
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
        res = self.model( batch[0][:,0], None, is_test=True )
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
        optimizer = optim.Adagrad( self.parameters(), lr=self.config['lr'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        max_epochs=1, 
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
           EarlyStopping(monitor="ndcg_10", patience=10, mode="max", min_delta=0.01)
        ],
        progress_bar_refresh_rate=0
    )

    model = ModelTrainer( config, dataset )

    trainer.fit( model )

def test_model( config : dict, checkpoint_dir : str, dataset ):
    model = ModelTrainer.load_from_checkpoint( config=config, checkpoint_path=os.path.join( checkpoint_dir, 'checkpoint' ), dataset=dataset )

    trainer = pl.Trainer()
    result = trainer.test( model )
    print( result )

    save_json = {
        'checkpoint_dir' : str( checkpoint_dir ),
        'result' : result[0]
    }

    with open('best_model_result.json','w') as f:
        json.dump( save_json, f )

def tune_model( relation : str ):
    ray.init( num_cpus=10 )
    dataset = ray.put( Dataset() )
    config = {
        # grid search parameter
        'num_latent' : tune.grid_search([ 8, 16, 32 ]),
        'num_group' : tune.grid_search([ i for i in range( 2, 22, 2 ) ]),

        # hopefully will find right parameter
        'prediction_margin' : tune.choice([ i for i in range( 1, 6 ) ]),
        'transition_margin' : tune.choice([ 1e-3, 5e-3, 1e-2, 5e-2, 1e-1 ]),
        'gamma' : tune.choice([ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ]),
        'batch_size' : tune.choice([ 32, 64, 128, 256, 512 ]),
        'lr' : tune.quniform( 1e-3, 1e-2, 1e-3 ),
        'alpha' : tune.quniform( 10, 200, 10 ),
        'beta' : tune.qrandint( 10, 100, 10 ),
        'min_lambda' : tune.quniform( 0.6, 0.8, 1e-2 ),
        'lambda' : tune.quniform(0.85,0.99,1e-2),
        'relation' : relation,
        'hr_k' : 20,
        'recall_k' : 20,
        'ndcg_k' : 100
    }

    scheduler = ASHAScheduler(
        max_t=256,
        grace_period=10,
        reduction_factor=2
    )

    reporter = CLIReporter( 
        parameter_columns=[ 'num_latent', 'gamma', 'num_group' ],
        metric_columns=[ 'hr_score', 'recall_score', 'ndcg_score'  ]
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 2 },
        metric='ndcg_score',
        mode='max',
        num_samples=5,
        verbose=1,
        config=config,
        progress_reporter=reporter,
        scheduler=scheduler,
        name=f'yelp_dataset_{relation}',
        local_dir=f"/data2/saito/yelp_relation_{relation}",
        keep_checkpoints_num=1, 
        checkpoint_score_attr='ndcg_score'
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )
if __name__ == '__main__':
    tune_model( 'BCat' )

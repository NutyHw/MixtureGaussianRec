import os
import json 
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model
from utilities.dataset.dataloader import Scheduler
from utilities.dataset.yelp_dataset import YelpDataset
from torch.utils.data import DataLoader, TensorDataset
from ndcg import ndcg
import torch.optim as optim
import pytorch_lightning as pl

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

# some hack from https://github.com/ray-project/ray/issues/10995
os.environ["SLURM_JOB_NAME"] = "bash"

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict ):
        super().__init__()
        self.config = config
        self.download_data()

        self.model = Model( self.dataset.n_users, self.dataset.n_items, self.reg_mat.shape[0], config['num_group'], config['num_latent'] )
        self.prediction_loss = nn.MarginRankingLoss( margin=config['prediction_margin'], reduction='mean' )
        self.transition_loss = nn.MarginRankingLoss( margin=config['transition_margin'], reduction='mean' )

        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']

    def download_data( self ):
        self.dataset = YelpDataset( self.config['relation_id'] )
        self.reg_mat = self.dataset.get_reg_mat()

    def train_dataloader( self ):
        return DataLoader( self.dataset, batch_size=self.config['batch_size'], num_workers=0 )

    def val_dataloader( self ):
        x = self.dataset.get_val()
        y = torch.zeros( ( x.shape[0] // 101, 101 ) )
        y[ :, 0 ] = 1
        y = y.reshape( -1, 1 )

        return DataLoader( TensorDataset( x, y ), batch_size=2048, shuffle=False, num_workers=0 )

    def test_dataloader( self ):
        x = self.dataset.get_test()
        y = torch.zeros( ( x.shape[0] // 101, 101 ) )
        y[ :, 0 ] = 1
        y = y.reshape( -1, 1 )

        return DataLoader( TensorDataset( x, y ), batch_size=2048, shuffle=False, num_workers=0 )

    def joint_loss( self, pos_result1, neg_result1, pos_result2, neg_result2 ):
        return torch.mean( torch.relu( - ( pos_result1 - neg_result1 + self.config['prediction_margin'] ) * ( pos_result2 - neg_result2 + self.config['transition_margin'] ) ) )

    def normal_regularization( self, p : torch.Tensor ):
        num_latent = p.shape[1] // 2
        mu = p[:,:num_latent]
        Sigma = p[:,num_latent:]

        return 0.5 * torch.sum( (
            torch.sum( mu ** 2, dim=-1 ) +\
            torch.sum( Sigma, dim=-1 ) -\
            num_latent -\
            torch.log( torch.prod( Sigma, dim=-1 ) )
    ) )

    def evaluate( self, true_rating, predict_rating ):
        _, top_k_indices = torch.topk( predict_rating, k=1, dim=1, largest=True )
        hr_1 = torch.mean( torch.gather( true_rating, dim=-1, index=top_k_indices ) )

        _, top_k_indices = torch.topk( predict_rating, k=10, dim=-1, largest=True )

        recall_10 = torch.nanmean( 
            torch.sum( torch.gather( true_rating, dim=1, index = top_k_indices ), dim=-1 )
        )

        ndcg_10 = torch.mean( ndcg( predict_rating, true_rating, [ 10 ] ) )

        return hr_1.item(), recall_10.item(), ndcg_10.item()

    def training_step( self, batch, batch_idx ):
        pos_interact, neg_interact = batch
        batch_size = self.config['batch_size']

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
        category_reg = F.kl_div( res['category_kg'], self.reg_mat.T[ item_idx ] )
        normal_loss = self.normal_regularization( res['distribution'][0] ) + self.normal_regularization( res['distribution'][1] )

        return l1_loss * self.alpha + l2_loss * self.beta + l3_loss + self.gamma * ( normal_loss + category_reg )

    def on_train_epoch_end( self ):
        self.alpha = max( self.alpha * self.config['lambda'], self.config['alpha'] * self.config['min_lambda'] )
        self.beta = max( self.beta * self.config['lambda'], self.config['beta'] * self.config['min_lambda'] )

    def on_validation_epoch_start( self ):
        self.predict_score = torch.zeros( ( 0, 1 ) )
        self.true_score = torch.zeros( ( 0, 1 ) )

    def validation_step( self, batch, batch_idx ):
        interact, y = batch
        res = self.model( interact[:,0], interact[:,1], is_test=True )
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
        self.predict_score = torch.vstack( ( self.predict_score, res['kg_prob'] ) )
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
        optimizer = optim.RMSprop( self.parameters(), lr=self.config['lr'] )
        return optimizer

def train_model( config, checkpoint_dir=None ):
    trainer = pl.Trainer(
        max_epochs=1, 
        limit_train_batches=1,
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
           )
        ]
    )

    model = ModelTrainer( config )

    trainer.fit( model )

def test_model( config : dict, checkpoint_dir : str ):
    model = ModelTrainer.load_from_checkpoint( config=config, checkpoint_path=os.path.join( checkpoint_dir, 'checkpoint' ) )
    # checkpoint = torch.load( os.path.join( checkpoint_dir, 'checkpoint' ) )
    # model = model.load_state_dict( checkpoint["state_dict"] )

    trainer = pl.Trainer()
    result = trainer.test( model )

    save_json = {
        'checkpoint_dir' : checkpoint_dir,
        'result' : result[0]
    }

    with open('best_model_result.json','w') as f:
        json.dump( save_json, f )

def tune_model( relation_id : int ):
    config = {
        # grid search parameter
        # 'num_latent' : tune.grid_search([ 4, 8, 16, 32 ]),
        # 'gamma' : tune.grid_search([ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ]),
        # 'num_group' : tune.grid_search([ 5 * i for i in range( 1, 11 ) ]),
        'num_latent' : tune.grid_search([ 4 ]),
        'gamma' : tune.grid_search([ 1e-5 ]),
        'num_group' : tune.grid_search([ 5 ]),

        # hopefully will find right parameter
        'batch_size' : tune.choice([ 128, 256, 512, 1024 ]),
        'lr' : tune.choice([ 0.05, 0.01, 0.005, 0.001 ]),
        'alpha' : 1,
        'beta' : tune.qrandint( 50, 1000, 50 ),
        'min_lambda' : tune.quniform( 0.6, 0.8, 5e-2 ),
        'prediction_margin' : tune.quniform( 0.1, 5, 0.1 ),
        'transition_margin' : tune.quniform( 0.1, 5, 0.1 ),
        'lambda' : tune.quniform( 0.7, 0.95, 1e-2 ),
        'relation_id' : relation_id
    }

    scheduler = ASHAScheduler(
        max_t=128,
        grace_period=5,
        reduction_factor=2
    )

    reporter = CLIReporter( 
        parameter_columns=[ 'num_latent', 'gamma', 'num_group' ],
        metric_columns=[ 'hr_1', 'recall_10', 'ndcg_10'  ]
    )

    analysis = tune.run( 
        train_model,
        resources_per_trial={ 'cpu' : 1 },
        metric='ndcg_10',
        mode='max',
        config=config,
        progress_reporter=reporter,
        scheduler=scheduler,
        name=f'yelp_relation_{relation_id}'
    )

    test_model( analysis.best_config, analysis.best_checkpoint )

if __name__ == '__main__':
    tune_model( 0 )

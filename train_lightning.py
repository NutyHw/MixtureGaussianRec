import sys
import os
import json 
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F
from models.model import GCN as Model
from torch.utils.data import DataLoader, TensorDataset
from ndcg import ndcg
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch_cluster import random_walk

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from utilities.dataset.dataloader import Scheduler
from utilities.dataset.yelp_dataset import YelpDataset as Dataset

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )
        self.n_users, self.n_items = self.dataset.n_users, self.dataset.n_items

        self.model = Model( self.n_users + self.n_items, self.config['num_latent'], self.config['num_hidden'], self.config['activation'] )
        self.edge_indices = self.dataset.train_interact.to( self.device )

    def train_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.n_users, self.n_items  ) ), batch_size=self.config['batch_size'],  collate_fn=self.samples )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( 1 ), batch_size=1 )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( 1 ), batch_size=1 )

    def samples( self, batch ):
        return self.pos_sample(batch), self.neg_sample(batch)

    def pos_sample( self, batch ):
        batch = batch.repeat( self.config['walks_per_node'] )
        rw = random_walk( self.dataset.train_interact[0], self.dataset.train_interact[1], batch, self.config['walk_length'], self.config['p'], self.config['q'])

        walks = []
        num_walks_per_rw = 1 + self.config['walk_length'] + 1 - self.config['context_size']

        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.config['context_size']])

        return torch.cat(walks, dim=0)

    def neg_sample( self, batch ):
        batch = batch.repeat(self.config['walks_per_node'] * self.config['neg_sample'] )

        rw = torch.randint(self.n_users + self.n_items,
                           (batch.size(0), self.config['walk_length']))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.config['walk_length'] + 1 - self.config['context_size']
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.config['context_size']])
        return torch.cat(walks, dim=0)

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
        pos_rw, neg_rw = batch
        start, rest = pos_rw[:,0], pos_rw[:,1:].contiguous()

        X = F.one_hot( self.n_users + self.n_items ).to( self.device, torch.float )

        embed = self.model( X, self.edge_indices )
        h_start  = embed[ start ].reshape( pos_rw.shape[0], 1, self.config['num_latent'] )
        h_rest  = embed[ rest.view( -1 ) ].reshape( pos_rw.shape[0], 1, self.config['num_latent'] )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = - torch.log(torch.sigmoid(out) + 1e-9).mean()

        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.model(start, self.edge_indices).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.model(rest.view(-1), self.edge_indices).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        neg_loss = -torch.log( torch.sigmoid( -out ) + 1e-9 ).mean()

        return pos_loss + neg_loss

    def validation_step( self, batch, batch_idx ):
        X, edge_indices, adj, user_user_sim, item_item_sim  = batch
        embed = self.model( X, edge_indices )
        user_embed, item_embed = embed[ : self.n_users ], embed[ self.n_users : ]
        pred_user_item_sim = torch.sigmoid( torch.mm( user_embed, item_embed.T ) )
        val_mask, true_y = self.dataset.get_val()
        pred_user_item_sim[ ~val_mask ] = -np.inf

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, pred_user_item_sim, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score,
        })

    def test_step( self, batch, batch_idx ):
        X, edge_indices, adj, user_user_sim, item_item_sim  = batch
        embed = self.model( X, edge_indices )
        user_embed, item_embed = embed[ : self.n_users ], embed[ self.n_users : ]
        pred_user_item_sim = torch.sigmoid( torch.mm( user_embed, item_embed.T ) )
        val_mask, true_y = self.dataset.get_test()
        pred_user_item_sim[ ~val_mask ] = -np.inf

        hr_score, recall_score, ndcg_score = self.evaluate( true_y, pred_user_item_sim, self.config['hr_k'], self.config['recall_k'], self.config['ndcg_k'] )

        self.log_dict({
            'hr_score' : hr_score,
            'recall_score' : recall_score,
            'ndcg_score' : ndcg_score,
        })

    def configure_optimizers( self ):
        optimizer = optim.SGD( self.parameters(), lr=self.config['lr'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        limit_train_epoches=1,
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
    ray.init( num_cpus=8, num_gpus=8 )
    dataset = ray.put( Dataset( relation ) )
    config = {
        # parameter to find
        'num_latent' : 64,
        'num_hidden' : 2,
        'activation' : 'relu'
        'lr' : 1e-3,
        'walks_per_node' : tune.grid_search([ i * 2 for i in range( 3, 11 ) ]),
        'walk_length' : tune.grid_search([ i * 10 for i in range( 3, 11 ) ]),
        'context_size' : tune.grid_search([ i * 2 for i in range( 4, 11 ) ]),

        # fix parameter
        'p' : 0.1,
        'q' : 0.1,
        'relation' : relation,
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
        metric='ndcg_score',
        mode='max',
        verbose=1,
        num_samples=1,
        config=config,
        scheduler=scheduler,
        name=f'non_colapse_yelp_dataset_{relation}_3',
        keep_checkpoints_num=2,
        local_dir=f"./",
        checkpoint_score_attr='ndcg_score',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )
if __name__ == '__main__':
    tune_population_based( sys.argv[1] )

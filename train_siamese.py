import sys
import os
from functools import partial

import torch
import torch.nn.functional as F
from models.model import SiameseModel as Model
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pytorch_lightning as pl

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from utilities.dataset.siamese import SiameseDataset as Dataset

os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

class ModelTrainer( pl.LightningModule ):
    def __init__( self, config : dict, dataset=None ):
        super().__init__()
        self.config = config
        self.dataset = ray.get( dataset )

        self.model = Model( config['n'], config['num_latent'], config['num_hidden'] )

    def train_dataloader( self ):
        anchor, pos_anchor, neg_anchor = self.dataset.samples( self.config['neg_sample'] )
        return DataLoader( TensorDataset( anchor, pos_anchor, neg_anchor ), batch_size=self.config['batch_size'], shuffle=True )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( 1 ) ) )

    def training_step( self, batch, batch_idx ):
        anchor, pos_anchor, neg_anchor = batch

        anchor_input = F.one_hot( anchor ).to( self.device, torch.float )
        pos_anchor_input = F.one_hot( pos_anchor ).to( self.device, torch.float )
        neg_anchor_input = F.one_hot( neg_anchor ).to( self.device, torch.float )

        anchor_embed = self.model( anchor_input )
        pos_embed = self.model( pos_anchor_input )
        neg_embed  = self.model( neg_anchor_input )

        pos_dist = torch.dist( anchor_embed, pos_embed, 2 )
        neg_dist = torch.dist( anchor_embed, neg_embed, 2 )

        return torch.mean( pos_dist ) + torch.mean( torch.clamp( 1 - neg_dist, min=0 ) )

    def validation_step( self, batch, batch_idx ):
        embed = self.model( F.one_hot( torch.arange( self.config['n'] ) ).to( self.device, torch.float ) )
        cosine = F.cosine_similarity( embed.unsqueeze( dim=1 ), embed.unsqueeze( dim=0 ), dim=-1 )
        affinity = self.dataset.affinity

        pos_sim = torch.mean( ( affinity - torch.eye( affinity.shape ) ) * cosine.cpu() )
        neg_sim = torch.mean( ( ( 1 - affinity ) * cosine.cpu ) )

        self.log_dict( { 'similarity' : pos_sim - neg_sim } )

    def configure_optimizers( self ):
        optimizer = optim.Adam( self.parameters(), lr=self.config['lr'] )
        return optimizer

def train_model( config, checkpoint_dir=None, dataset=None ):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=20,
        callbacks=[
            TuneReportCheckpointCallback( {
                'similarity' : 'similarity'
            },
            on='validation_end',
            filename='checkpoint'
           ),
        ]
        check_val_every_n_epoch=5,
        progress_bar_refresh_rate=0
    )

    if checkpoint_dir:
        config["resume_from_checkpoint"] = os.path.join(
        checkpoint_dir, "checkpoint"
    )
    model = ModelTrainer( config, dataset )

    trainer.fit( model )

def tune_population_based( file ):
    ray.init( num_cpus=8, num_gpus=8 )
    dataset = ray.put( Dataset( './process_datasets/yelp_siamese/UCom_affinity.npy' ) )
    config = {
        # parameter to find
        'num_latent' : 64,
        'num_hidden' : 32,
        'neg_sample' : tune.grid_search([ i * 2 for i in range( 1, 6 ) ]),
        'n' : 7981,
        'lr' : tune.grid_search([ 1e-4, 5e-3, 1e-3, 5e-2, 1e-2 ]),
    }

    scheduler = ASHAScheduler(
        grace_period=10,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 1, 'gpu' : 1 },
        metric='similarity',
        mode='max',
        verbose=1,
        num_samples=1,
        config=config,
        scheduler=scheduler,
        keep_checkpoints_num=2,
        local_dir=f"./",
        checkpoint_score_attr='similarity',
    )

    test_model( analysis.best_config, analysis.best_checkpoint, dataset )
if __name__ == '__main__':
    tune_population_based( sys.argv[1] )

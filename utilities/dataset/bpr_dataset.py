import sys
import os
import random
from collections import defaultdict
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class YelpDataset( Dataset ):
    def __init__( self, process_dataset, i, dataset_type='train' ):
        assert dataset_type in [ 'train', 'val', 'test' ]
        self.dataset_type = dataset_type

        self.dataset = torch.load( os.path.join( process_dataset, f'yelp_dataset_{i}.pt' ) )
        self.n_users, self.n_items = self.dataset['train_adj'].shape
        self.create_interact()

    def create_interact( self ):
        num_samples = torch.sum( self.dataset[ 'train_adj' ] > 0, dim=-1 ).reshape( -1, 1 )
        neg_adj = ( 1 - ( self.dataset[ 'train_adj' ] > 0 ).to( torch.float ) )

        pos_interact = ( self.dataset['train_adj'] > 0 ).nonzero()
        neg_interact = torch.zeros( ( 0, 2 ), dtype=torch.long )

        for idx in range( num_samples.shape[0] ):
            item_idx = torch.multinomial( neg_adj[idx], num_samples=int( num_samples[idx].item() ) ).unsqueeze( dim=0 )
            user_idx = torch.full( ( 1, item_idx.shape[1] ), idx )
            neg_interact = torch.vstack( ( neg_interact, torch.vstack( ( user_idx, item_idx ) ).T ) )

        self.pos_interact = pos_interact
        self.neg_interact = neg_interact
        
        print( self.pos_interact.shape )
        print( self.neg_interact.shape )

    def val_interact( self ):
        val_data = self.dataset['val_dataset']
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return val_data, y

    def test_interact( self ):
        test_data = self.dataset['test_dataset']
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return test_data, y

    def __len__( self ):
        return self.pos_interact.shape[0]

    def __getitem__( self, idx ):
        return self.pos_interact[ idx ], self.neg_interact[ idx ]

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/', 'seen_val_test' )
    print( dataset[ 2896 ] )
    print( dataset.val_interact() )

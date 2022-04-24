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
        self.create_input( process_dataset )
        self.create_interact()

    def create_input( self, process_dataset ):
        UU, UCom, BCat, BCity = None, None, None,None

        with open( os.path.join( process_dataset, 'attribute.npy' ), 'rb' ) as f:
            arr = np.load( f )
            UU = self.normalize( torch.from_numpy( arr['UU'] ).to( torch.float ) )
            UCom = self.normalize( torch.from_numpy( arr['UCom'] ).to( torch.float ) )
            BCat = self.normalize( torch.from_numpy( arr['BCat'] ).to( torch.float ) )
            BCity = self.normalize( torch.from_numpy( arr['BCity'] ).to( torch.float ) )

        self.n_users, self.n_items = self.dataset['train_adj'].shape
        user_one_hot = F.one_hot( torch.arange( self.n_users ) ).to( torch.float )
        item_one_hot = F.one_hot( torch.arange( self.n_items ) ).to( torch.float )

        self.user_input = torch.hstack( ( UU, UCom, user_one_hot ) )
        self.item_input = torch.hstack( ( BCat, BCity, item_one_hot ) )

    def create_interact( self ):
        train_adj_mat = self.dataset[ 'train_adj' ]
        pos_interact = train_adj_mat.nonzero()
        neg_interact = ( 1 - ( train_adj_mat > 0 ).to( torch.int ) ).nonzero()

        sample_neg_idx = torch.randint( neg_interact.shape[0], ( pos_interact.shape[0], ) )
        sample_neg_interact = neg_interact[ sample_neg_idx ]

        Y = torch.vstack( ( torch.ones( ( pos_interact.shape[0], 1 ) ), torch.zeros( ( neg_interact.shape[0], 1 ) ) ) )

        self.interact = torch.vstack( ( pos_interact, sample_neg_interact ) )
        self.Y = Y

    def val_interact( self ):
        val_data = self.dataset['val_dataset']
        user_idx = torch.arange( self.n_users ).reshape( -1, 1 ).tile( 1, 101 ).reshape( -1, 1 )
        item_idx = val_data.reshape( -1, 1 )
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return torch.hstack( ( user_idx, item_idx ) ), y

    def test_interact( self ):
        val_data = self.dataset['test_dataset']
        user_idx = torch.arange( self.n_users ).reshape( -1, 1 ).tile( 1, 101 ).reshape( -1, 1 )
        item_idx = val_data.reshape( -1, 1 )
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return torch.hstack( ( user_idx, item_idx ) ), y

    def normalize( self, mat ):
        return mat / torch.sum( mat, dim=-1 ).reshape( -1, 1 )

    def __len__( self ):
        return self.interact.shape[0]

    def __getitem__( self, idx ):
        return self.user_input[ self.interact[idx][0] ], self.item_input[ self.interact[idx][1] ], self.Y[idx]

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/', 0 )
    print( dataset.val_interact() )
    print( dataset.test_interact() )

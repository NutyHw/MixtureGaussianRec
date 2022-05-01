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
    def __init__( self, process_dataset, i, neg_size = 10 ):
        self.dataset = torch.load( os.path.join( process_dataset, f'yelp_dataset_{i}.pt' ) )
        self.create_input( process_dataset )
        self.create_interact( neg_size )

    def create_input( self, process_dataset ):
        UU, UCom, BCat, BCity = None, None, None,None

        with open( os.path.join( process_dataset, 'attribute_leiden.npz' ), 'rb' ) as f:
            arr = np.load( f )
            UU = self.normalize( torch.from_numpy( arr['UUCom'] ).to( torch.float ) )
            UCom = self.normalize( torch.from_numpy( arr['UCom'] ).to( torch.float ) )
            BCat = self.normalize( torch.from_numpy( arr['BCat'] ).to( torch.float ) )
            BCity = self.normalize( torch.from_numpy( arr['BCity'] ).to( torch.float ) )

        self.n_users, self.n_items = self.dataset['train_adj'].shape
        print( self.n_users, self.n_items )
        user_one_hot = F.one_hot( torch.arange( self.n_users ) ).to( torch.float )
        item_one_hot = F.one_hot( torch.arange( self.n_items ) ).to( torch.float )

        self.user_input = torch.hstack( ( UU, UCom, user_one_hot ) )
        self.item_input = torch.hstack( ( BCat, BCity, item_one_hot ) )

    def create_interact( self, neg_size : int ):
        train_adj_mat = self.dataset[ 'train_adj' ]
        pos_interact = ( train_adj_mat > 0 ).nonzero()
        neg_interact = torch.randint( self.n_items, ( pos_interact.shape[0], 10 ) )

        self.pos_interact = pos_interact
        self.neg_interact = neg_interact

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
        return self.pos_interact.shape[0]

    def __getitem__( self, idx ):
        return self.user_input[ self.pos_interact[idx][0] ], self.item_input[ self.pos_interact[idx][1] ], self.item_input[ self.neg_interact[ idx] ]
if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/', '0' )
    user_input, pos_item_input, neg_item_input = dataset[0] 

    print( user_input.shape, pos_item_input.shape, neg_item_input.shape )

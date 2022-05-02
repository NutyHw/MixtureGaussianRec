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

        arr = torch.load( os.path.join( process_dataset, 'metapath_input.pt' ) )
        self.user_weight = arr['user_weight']
        self.item_weight = arr['item_weight'] 
        self.norm_log_gaussian = arr['norm_log_gauss_mat']

        train_adj = ( self.dataset['train_adj'] > 0 ).to( torch.float )
        self.true_y = train_adj / torch.sum( train_adj, dim=-1 ).reshape( -1, 1 )

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

    def __len__( self ):
        return self.user_gaussian.shape[0]

    def __getitem__( self, idx ):
        return self.user_weight[ idx ], self.item_weight, self.norm_log_gaussian, self.true_y[ idx ]

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/', 'cold_start' )
    user_weight, item_weight, norm_log_gaussian, true_y = dataset[0] 
    print( user_weight, item_weight, norm_log_gaussian, true_y )

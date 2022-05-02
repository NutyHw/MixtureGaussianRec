import sys
import os
import random
from collections import defaultdict
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class YelpDataset( Dataset ):
    def __init__( self, process_dataset, i, neg_size = 10 ):
        self.dataset = torch.load( os.path.join( process_dataset, f'yelp_dataset_{i}.pt' ) )
        self.n_users, self.n_items = self.dataset['train_adj'].shape

        arr = torch.load( os.path.join( process_dataset, 'metapath_input.pt' ) )
        self.user_weight = arr['user_weight'].to( torch.float )
        self.item_weight = arr['item_weight'].to( torch.float )
        self.norm_log_gaussian = arr['norm_log_gauss_mat'].to( torch.float )

        self.n_users, self.n_items = self.dataset['train_adj'].shape

        with open( os.path.join( process_dataset, 'embedding.npz' ), 'rb' ) as f:
            embedding = np.load( f )
            self.user_embed =  torch.from_numpy( embedding['user'] ).to( torch.float )
            self.item_embed =  torch.from_numpy( embedding['item'] ).to( torch.float )

        train_adj = ( self.dataset['train_adj'] > 0 ).to( torch.float )
        self.true_y = train_adj / torch.sum( train_adj, dim=-1 ).reshape( -1, 1 )

    def val_interact( self ):
        val_data = self.dataset['val_dataset']
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return val_data, y

    def test_interact( self ):
        val_data = self.dataset['test_dataset']
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return val_data, y

    def __len__( self ):
        return self.n_users

    def __getitem__( self, idx ):
        return self.user_embed[ idx ], self.item_embed, self.user_weight[ idx ], self.item_weight, self.norm_log_gaussian, self.true_y[ idx ]

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/', 'cold_start' )
    #loader = DataLoader( dataset, batch_size=256 )
    #for idx, batch in enumerate( loader ):
    #    user_weight, item_weight, log_gauss_mat, true_prob = batch
    #    print( user_weight.shape, item_weight[0].shape ,log_gauss_mat[0].shape, true_prob.shape )

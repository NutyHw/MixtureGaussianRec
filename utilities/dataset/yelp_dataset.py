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
    def __init__( self, process_dataset ):
        with open( os.path.join( process_dataset, 'train_mask' ), 'rb' ) as f:
            self.train_mask = torch.from_numpy( np.load( f ) ) > 0
        with open( os.path.join( process_dataset, 'val_mask' ), 'rb' ) as f:
            self.val_mask = torch.from_numpy( np.load( f ) ) > 0
        with open( os.path.join( process_dataset, 'test_mask' ), 'rb' ) as f:
            self.test_mask = torch.from_numpy( np.load( f ) ) > 0

        with open( os.path.join( process_dataset, 'UB.npy' ), 'rb' ) as f:
            self.ub = torch.from_numpy( np.load( f ) )

        self.train_adj_mat = self.train_mask * self.ub
        #self.filter_coldstart_item()

        self.n_users, self.n_items = self.train_adj_mat.shape

        self.samples()

    def get_stats( self ):
        return self.train_adj_mat.shape[0], self.train_adj_mat.shape[1]

    def filter_coldstart_item( self ):
        item_mask = torch.sum( self.train_adj_mat , dim=0 ) > 0

        self.ub = self.ub[ :, item_mask ]
        self.train_mask = self.train_mask[ :, item_mask ] > 0
        self.val_mask = self.val_mask[ :, item_mask ] > 0
        self.test_mask = self.test_mask[ :, item_mask ] > 0
        self.train_adj_mat = self.train_adj_mat[ :, item_mask ]

        self.item_mask = item_mask

    def __len__( self ):
        return self.pos_interact.shape[0]

    def __getitem__( self, idx ):
        return self.pos_interact[ idx ], self.neg_interact[ idx ]

    def get_val( self ):
        return self.val_mask, self.ub * self.val_mask

    def get_test( self ):
        return self.test_mask, self.ub * self.test_mask

    def samples( self ):
        user_count = torch.sum( self.train_adj_mat, dim=-1 ).to( torch.int ).tolist()
        neg_adj = 1 - ( self.train_adj_mat > 0 ).to( torch.float )
        neg_adj = neg_adj / torch.sum( neg_adj, dim=-1 ).reshape( -1, 1 )

        neg_interact = torch.zeros( ( 0, 2 ), dtype=torch.long )

        for idx, size in enumerate( user_count ):
            chosen_items = torch.tensor( np.random.choice( self.n_items, size, p=neg_adj[idx].numpy() ) )
            users = torch.full( chosen_items.shape, idx )
            interact = torch.vstack( ( users, chosen_items ) ).T
            neg_interact = torch.vstack( ( neg_interact, interact ) )

        self.pos_interact = self.train_adj_mat.nonzero()
        self.neg_interact = neg_interact

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/train_ratio_0.6/' )
    pos, neg = dataset[0]
    print( pos, neg )

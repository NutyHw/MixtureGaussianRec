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
        with open( os.path.join( process_dataset, 'dataset.npz' ), 'rb' ) as f:
            arr = np.load( f )
            self.ub = torch.from_numpy( arr['ub'] )
            self.item_attribute = torch.from_numpy( arr['bcat'] )
            self.train_mask = torch.from_numpy( arr['train_mask'] )
            self.val_mask = torch.from_numpy( arr['val_mask'] )
            self.test_mask = torch.from_numpy( arr['test_mask'] )

        self.train_adj_mat = self.train_mask * self.ub
        self.n_users, self.n_items = self.train_adj_mat.shape

        self.user_input = F.one_hot( torch.arange( self.n_users ) ).to( torch.float )
        item_one_hot = F.one_hot( torch.arange( self.n_items ) ).to( torch.float )
        self.item_input = torch.hstack( ( item_one_hot, self.item_attribute ) ).to( torch.float )  

        #self.filter_coldstart_item()
        #self.samples()

    def __len__( self ):
        return self.n_users

    def __getitem__( self, idx ):
        return self.user_input[ idx ], self.item_input, self.train_adj_mat[ idx ]

    def get_item_attribute( self ):
        return self.item_attribute

    def get_val( self ):
        return self.val_mask, self.ub * self.val_mask

    def get_test( self ):
        return self.test_mask, self.ub * self.test_mask

    def filter_coldstart_item( self ):
        item_mask = torch.sum( self.train_adj_mat , dim=0 ) > 0

        self.train_mask = self.train_mask[ :, item_mask ]
        self.val_mask = self.val_mask[ :, item_mask ]
        self.test_mask = self.test_mask[ :, item_mask ]
        self.train_adj_mat = self.train_adj_mat[ :, item_mask ]
        self.item_input = self.item_input[ :, item_mask ]

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
    dataset = YelpDataset( './yelp_dataset/train_ratio_0.8/' )
    user_input, item_input, train_adj = dataset[0]
    print( user_input.shape, item_input.shape, train_adj.shape )

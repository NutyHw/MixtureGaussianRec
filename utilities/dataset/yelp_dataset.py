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
    def __init__( self, process_dataset, k ):
        self.k = k
        with open( os.path.join( process_dataset, 'dataset.npz' ), 'rb' ) as f:
            arr = np.load( f )
            self.ub = torch.from_numpy( arr['ub'] )
            self.item_attribute = torch.from_numpy( arr['bcat'] ).to( torch.float )
            self.train_mask = torch.from_numpy( arr['train_mask'] ) > 0
            self.val_mask = torch.from_numpy( arr['val_mask'] ) > 0
            self.test_mask = torch.from_numpy( arr['test_mask'] ) > 0

        self.train_adj_mat = self.train_mask * self.ub
        self.n_users, self.n_items = self.train_adj_mat.shape

        self.user_input = F.one_hot( torch.arange( self.n_users ) ).to( torch.float )
        item_one_hot = F.one_hot( torch.arange( self.n_items ) ).to( torch.float )
        self.item_input = torch.hstack( ( item_one_hot, self.item_attribute ) ).to( torch.float )  

        #self.filter_coldstart_item()
        self.samples()

    def __len__( self ):
        return self.pos_interact.shape[0]

    def __getitem__( self, idx ):
        negative_idx = torch.randint( self.n_items, ( self.k,  ) )
        return self.user_input[ self.pos_interact[ idx ][0] ], \
            self.item_input[ self.pos_interact[ idx ][1] ], \
            self.item_input[ negative_idx ]

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
        self.pos_interact = self.train_adj_mat.nonzero()

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/train_ratio_0.8/', 5 )
    user_input, item_input, train_adj = dataset[0]
    print( dataset.get_item_attribute().shape )
    print( user_input.shape, item_input.shape, train_adj.shape )

import random
from collections import defaultdict
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

random.seed(7)

class YelpDataset( Dataset ):
    def __init__( self ):
        self.sample_size = 5
        self.UB, self.BCat, self.BCity = self.load_dataset()
        self.BCat, self.BCity = self.preprocess_relation( self.BCat ), self.preprocess_relation( self.BCity )
        self.n_users, self.n_items = self.UB.shape[0], self.UB.shape[1]

        self.pos_train_interact, pos_val_interact, pos_test_interact = self.train_test_val_split( self.UB )
        self.filter_cold_start( self.pos_train_interact, pos_val_interact, pos_test_interact )
        self.val_mask, self.val_score, self.test_mask, self.test_score = self.create_mask()

    def __len__( self ):
        return self.n_users

    def __getitem__( self, idx ):
        return idx, self.train_adj_mat[ idx ], None

    def get_val( self ):
        return self.val_mask, self.val_score

    def get_test( self ):
        return self.test_mask, self.test_score

    def preprocess_relation( self, relation  ):
        val, indices = relation.data, np.vstack((relation.row, relation.col))
        shape = relation.shape
        i = torch.LongTensor(indices)
        v = torch.FloatTensor( val )

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def get_reg_mat( self, relation : str ):
        if relation == 'BCat':
            return self.BCat
        elif relation == 'BCity':
            return self.BCity

    def load_dataset( self ):
        mat = scipy.io.loadmat( './process_datasets/yelp.mat' )
        UB, UU, UCom, BCat, BCity = (x.tocoo() for x in list(mat['relation'][0]))

        return UB, BCat, BCity

    def create_pytorch_interact( self, user_id, all_items ):
        user = torch.full( ( 1, len( all_items ) ), user_id, dtype=torch.long )
        all_items = torch.tensor( all_items, dtype=torch.long )

        return torch.vstack( ( user, all_items ) ).T

    def train_test_val_split( self, UB ):
        user_biz = defaultdict(list)

        for u, b in zip(UB.row, UB.col):
            user_biz[u].append(b)

        pos_train_interact = torch.zeros( ( 0, 2 ), dtype=torch.long )
        pos_val_interact = torch.zeros( ( 0, 2  ), dtype=torch.long )
        pos_test_interact = torch.zeros( ( 0, 2 ), dtype=torch.long )

        for u in user_biz:
            if len(user_biz[u]) >= 5:
                val = int( len(user_biz[u]) * 0.6 )
                test = int( len(user_biz[u]) * 0.8 )

                random.shuffle(user_biz[u])

                pos_train_interact = torch.vstack( ( pos_train_interact, self.create_pytorch_interact( u, user_biz[u][:val] ) ) )
                pos_val_interact = torch.vstack( ( pos_val_interact, self.create_pytorch_interact( u, user_biz[u][val:test] ) ) )
                pos_test_interact = torch.vstack( ( pos_test_interact, self.create_pytorch_interact( u, user_biz[u][test:] ) ) )

        return pos_train_interact, pos_val_interact, pos_test_interact

    def apply_mask_to_interact( self, user_mask, item_mask, interact ):
        adj_mat = torch.zeros( ( self.n_users, self.n_items ) )
        adj_mat[ interact[:,0], interact[:,1] ] = 1

        return adj_mat[ user_mask][ :, item_mask ].nonzero()

    def filter_cold_start( self, pos_train_interact, pos_val_interact, pos_test_interact ):
        train_adj_mat = torch.zeros( ( self.n_users, self.n_items ) )
        train_adj_mat[ pos_train_interact[:,0], pos_train_interact[:,1] ] = 1

        item_mask = torch.sum( train_adj_mat, dim=0 ) > 0
        train_adj_mat = train_adj_mat[ :, item_mask ]
        user_mask = torch.sum( train_adj_mat, dim=-1 ) > 0
        train_adj_mat = train_adj_mat[ user_mask ]

        self.train_adj_mat = train_adj_mat

        self.pos_train_interact = train_adj_mat.nonzero()
        self.pos_val_interact = self.apply_mask_to_interact( user_mask, item_mask, pos_val_interact )
        self.pos_test_interact = self.apply_mask_to_interact( user_mask, item_mask, pos_test_interact )

        self.n_users, self.n_items = train_adj_mat.shape[0], train_adj_mat.shape[1]

    def create_mask( self ):
        val_mask = torch.ones( ( self.n_users, self.n_items ) )
        test_mask = torch.ones( ( self.n_users, self.n_items ) )

        val_scores = torch.zeros( ( self.n_users, self.n_items ) )
        test_scores = torch.zeros( ( self.n_users, self.n_items ) )

        val_mask[ self.pos_train_interact[:,0], self.pos_train_interact[:,1] ] = 0
        val_mask[ self.pos_test_interact[:,0], self.pos_test_interact[:,1] ] = 0

        test_mask[ self.pos_train_interact[:,0], self.pos_train_interact[:,1] ] = 0
        test_mask[ self.pos_val_interact[:,0], self.pos_val_interact[:,1] ] = 0

        val_scores[ self.pos_val_interact[:,0], self.pos_val_interact[:,1] ] = 1
        test_scores[ self.pos_test_interact[:,0], self.pos_test_interact[:,1] ] = 1

        return val_mask > 0, val_scores, test_mask > 0, test_scores

if __name__ == '__main__':
    dataset = YelpDataset()
    print( dataset[0] )

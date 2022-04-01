import os
import random
from collections import defaultdict
import numpy as np
import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset

random.seed( 7 )

class YelpDataset( Dataset ):
    def __init__( self, relation : str, is_preprocess=False ):
        if not is_preprocess:
            self.process_dir = './process_datasets/yelp/'
            self.relation = relation
            self.load_data()
            self.samples()
        else:
            self.UB, self.UU, self.UCom, self.BCat, self.BCity = self.load_dataset()
            self.BCat, self.BCity, self.UU, self.UCom = self.preprocess_relation( self.BCat ), self.preprocess_relation( self.BCity ), self.preprocess_relation( self.UU ), self.preprocess_relation( self.UCom )

            self.n_users, self.n_items = self.UB.shape[0], self.UB.shape[1]
            self.pos_train_interact, pos_val_interact, pos_test_interact = self.train_test_val_split( self.UB )
            self.filter( self.pos_train_interact, pos_val_interact, pos_test_interact )
            self.val_mask, self.val_score, self.test_mask, self.test_score = self.create_mask()
            self.save_data()

    def __len__( self ):
        return self.pos_interact.shape[0]

    def __getitem__( self, idx ):
        return self.pos_interact[ idx ], self.neg_interact[ idx ], self.weight[ idx ]

    def get_val( self ):
        return self.test_mask, self.test_score

    def get_test( self ):
        return self.test_mask, self.test_score

    def load_data( self ):
        dataset = torch.load( os.path.join( self.process_dir, 'dataset.pt' ) )
        train_adj_mat = dataset['train_adj_mat']

        train_adj_mat = train_adj_mat / torch.sum( train_adj_mat, dim=-1 ).reshape( -1, 1 )
        self.train_adj_mat = train_adj_mat
        self.val_mask = dataset['val_mask']
        self.val_score = dataset['val_score']
        self.test_mask = dataset['test_mask']
        self.test_score = dataset['test_score']

        metapath = torch.load( os.path.join( self.process_dir, 'metapath.pt' ) )
        metapath = metapath[ self.relation ] + 1e-6
        self.metapath = metapath / torch.sum( metapath, dim=-1 ).reshape( -1, 1 )
        self.n_users, self.n_items = self.train_adj_mat.shape

    def save_data( self ):
        torch.save( { 
            'train_adj_mat' : self.train_adj_mat,
            'val_mask' : self.val_mask,
            'val_score' : self.val_score,
            'test_mask' : self.test_mask,
            'test_score' : self.test_score
        }, 'dataset.pt' )

        torch.save( {
            'UU' : self.UU,
            'UCom' : self.UCom,
            'BCat' : self.BCat,
            'BCity' : self.BCity
        }, 'metapath.pt' )

    def preprocess_relation( self, relation  ):
        val, indices = relation.data, np.vstack((relation.row, relation.col))
        shape = relation.shape
        i = torch.LongTensor(indices)
        v = torch.FloatTensor( val )

        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    def get_reg_mat( self ):
        return self.metapath

    def load_dataset( self ):
        mat = scipy.io.loadmat( './process_datasets/yelp.mat' )
        UB, UU, UCom, BCat, BCity = (x.tocoo() for x in list(mat['relation'][0]))

        return UB, UU, UCom, BCat, BCity

    def report_stats( self, UB ):
        val, indices = UB.data, np.vstack((UB.row, UB.col))
        shape = UB.shape
        i = torch.LongTensor(indices)
        v = torch.FloatTensor( val )

        adj = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        torch.save( adj, 'UB_before_preprocess.pt' )

    def create_pytorch_interact( self, user_id, all_items ):
        user = torch.full( ( 1, len( all_items ) ), user_id, dtype=torch.long )
        all_items = torch.tensor( all_items, dtype=torch.long )

        return torch.vstack( ( user, all_items ) ).T

    def train_test_val_split( self, UB ):
        user_biz = defaultdict(list)
        item_biz = defaultdict(list)

        for u, b in zip(UB.row, UB.col):
            user_biz[u].append(b)

        pos_train_interact = torch.zeros( ( 0, 2 ), dtype=torch.long )
        pos_val_interact = torch.zeros( ( 0, 2  ), dtype=torch.long )
        pos_test_interact = torch.zeros( ( 0, 2 ), dtype=torch.long )

        for u in user_biz:
            if len(user_biz[u]) >= 10:
                val = int( len(user_biz[u]) * 0.8 )
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

    def filter( self, pos_train_interact, pos_val_interact, pos_test_interact ):
        train_adj_mat = torch.zeros( ( self.n_users, self.n_items ) )
        val_adj_mat = torch.zeros( ( self.n_users, self.n_items ) )
        test_adj_mat = torch.zeros( ( self.n_users, self.n_items ) )

        train_adj_mat[ pos_train_interact[:,0], pos_train_interact[:,1] ] = 1
        val_adj_mat[ pos_val_interact[:,0], pos_val_interact[:,1] ] = 1
        test_adj_mat[ pos_test_interact[:,0], pos_test_interact[:,1] ] = 1

        # filter_item_less_than_10 = torch.sum( train_adj_mat + val_adj_mat + test_adj_mat, dim=0 ) > 9
        # train_adj_mat = train_adj_mat[ :, filter_item_less_than_10 ]
        # val_adj_mat = val_adj_mat[ :, filter_item_less_than_10 ]
        # test_adj_mat = test_adj_mat[ :, filter_item_less_than_10 ]
        # self.BCat = self.BCat[ filter_item_less_than_10 ]
        # self.BCity = self.BCity[ filter_item_less_than_10 ]

        train_item_mask = torch.sum( train_adj_mat, dim=0 ) > 0
        # val_item_mask = torch.sum( val_adj_mat, dim=0 ) > 0
        test_item_mask = torch.sum( test_adj_mat, dim=0 ) > 0

        item_mask = train_item_mask * test_item_mask
        train_adj_mat = train_adj_mat[ :, item_mask ]
        val_adj_mat = val_adj_mat[ :, item_mask ]
        test_adj_mat = test_adj_mat[ :, item_mask ]

        train_user_mask = torch.sum( train_adj_mat, dim=1 ) > 0
        val_user_mask = torch.sum( val_adj_mat, dim=1 ) > 0
        test_user_mask = torch.sum( test_adj_mat, dim=1 ) > 0

        user_mask = train_user_mask * test_user_mask
        train_adj_mat = train_adj_mat[ user_mask ]
        val_adj_mat = val_adj_mat[ user_mask ]
        test_adj_mat = test_adj_mat[ user_mask ]

        self.BCat = self.BCat[ item_mask ]
        self.BCity = self.BCity[ item_mask ]
        self.UU = self.UU[ user_mask ]
        self.UCom = self.UCom[ user_mask ]

        self.train_adj_mat = train_adj_mat

        self.pos_train_interact = train_adj_mat.nonzero()
        self.pos_val_interact = val_adj_mat.nonzero()
        self.pos_test_interact = test_adj_mat.nonzero()

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

    def samples( self ):
        user_count = torch.sum( self.train_adj_mat, dim=-1 ).to( torch.int ).tolist()
        neg_adj = 1 - self.train_adj_mat
        neg_adj = neg_adj / torch.sum( neg_adj, dim=-1 ).reshape( -1, 1 )

        neg_interact = torch.zeros( ( 0, 2 ), dtype=torch.long )
        for idx, size in enumerate( user_count ):
            chosen_items = torch.tensor( np.random.choice( self.n_items, size, p=neg_adj[idx].numpy() ) )
            users = torch.full( chosen_items.shape, idx )
            interact = torch.vstack( ( users, chosen_items ) ).T
            neg_interact = torch.vstack( ( neg_interact, interact ) )

        self.pos_interact = self.train_adj_mat.nonzero()
        self.neg_interact = neg_interact
        self.weight = self.train_adj_mat[ self.pos_interact[:,0], self.pos_interact[:,1] ]


if __name__ == '__main__':
    dataset = YelpDataset( 'UCom', is_preprocess=False )
    print( dataset[0] )

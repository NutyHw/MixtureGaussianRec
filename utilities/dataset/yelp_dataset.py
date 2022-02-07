import os
import torch
from torch.utils.data import Dataset
from filelock import FileLock

class YelpDataset( Dataset ):
    def __init__( self, r : int ):
        dataset_dir = '../../process_datasets/yelp2018/'
        self.relation_dir = os.path.join( dataset_dir, 'relation_mat' )
        self.train_adj_mat = torch.load( os.path.join( dataset_dir, 'train_adj_mat.pt' ) )

        self.n_users = self.train_adj_mat.shape[0]
        self.n_items = self.train_adj_mat.shape[1]

        self.load_dataset( r )
        self.sampling()

    def __len__( self ):
        return self.pos_train.shape[0]

    def __getitem__( self, idx ):
        return self.pos_train[ idx ], self.neg_train[ idx ]

    def get_val( self ):
        return self.relation_dataset[ 'val_interact'  ]

    def get_test( self ):
        return self.relation_dataset[ 'test_interact' ]

    def get_reg_mat( self ):
        return self.interact_mapper

    def load_dataset( self, r ):
        relation_dataset = dict()

        relation_dataset = torch.load( os.path.join( self.relation_dir, str(r), 'relation_dataset.pt' ) )
        entity_mapper = torch.load( os.path.join( self.relation_dir, str(r), 'entity_mapper.pt' ) )
        interact_mapper = torch.load( os.path.join( self.relation_dir, str(r), 'interact_mapper.pt' ) )

        self.relation_dataset = relation_dataset
        self.entity_mapper = entity_mapper
        self.interact_mapper = interact_mapper

    def sampling( self ):
        train_mask = self.relation_dataset[ 'train_mask' ]

        relation_adj_mat = self.train_adj_mat * train_mask
        relation_neg_adj_mat = ( 1 - self.train_adj_mat ) * train_mask

        user_mask = torch.sum( relation_adj_mat, dim=-1 ) > 0

        num_samples = torch.sum( relation_adj_mat, dim=-1 ).tolist()
        neg_train = torch.zeros( ( 0, 2 ) )

        for uid in torch.arange( self.train_adj_mat.shape[0] )[ user_mask ].tolist():
            neg_items = torch.multinomial( relation_neg_adj_mat[ uid ], num_samples=int( num_samples[ uid ] ) ).reshape( -1, 1 )
            user_idx = torch.full( ( int( num_samples[ uid ] ), 1 ), uid, dtype=torch.long )
            neg_train = torch.vstack( ( neg_train, torch.hstack( ( user_idx, neg_items ) ) ) )

        self.pos_train, self.neg_train = relation_adj_mat.nonzero(), neg_train

if __name__ == '__main__':
    dataset = YelpDataset( 1 )
    print( dataset[0] )

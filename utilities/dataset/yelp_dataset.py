import os
import torch
from torch.utils.data import Dataset
from filelock import FileLock

class YelpDataset( Dataset ):
    def __init__( self, r : int ):
        dataset_dir = '/Users/nuttupoomsaitoh/Desktop/class/seminar/PantipRec/process_datasets/yelp2018/'
        self.relation_dir = os.path.join( dataset_dir, 'relation_mat' )
        self.train_adj_mat = torch.load( os.path.join( dataset_dir, 'train_adj_mat.pt' ) )

        self.n_users = self.train_adj_mat.shape[0]
        self.n_items = self.train_adj_mat.shape[1]

        self.sample_size = 20
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

        pos_items = torch.multinomial( relation_adj_mat[ user_mask, : ], num_samples=self.sample_size, replacement=True )
        neg_items = torch.multinomial( relation_neg_adj_mat[ user_mask, : ], num_samples=self.sample_size, replacement=True )

        user_idx = torch.arange( self.train_adj_mat.shape[0] )[ user_mask ].reshape( -1, 1 ).tile( 1, self.sample_size ).reshape( -1, 1 )

        self.pos_train, self.neg_train = torch.hstack( ( user_idx, pos_items.reshape( -1, 1 ) ) ), torch.hstack( ( user_idx, neg_items.reshape( -1, 1 ) ) )

if __name__ == '__main__':
    dataset = YelpDataset( 0 )
    print( len( dataset ) )
    print( dataset[0], dataset[1] )
    print( dataset.get_reg_mat().shape )
    print( dataset.get_test().shape )
    print( dataset.get_val().shape )

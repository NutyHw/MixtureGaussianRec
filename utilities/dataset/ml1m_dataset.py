import os
import torch
from torch.utils.data import Dataset

class Ml1mDataset( Dataset ):
    def __init__( self, r : int ):
        dataset_dir = './process_datasets/ml-1m/'
        self.relation_dir = os.path.join( dataset_dir, 'relation_mat' )
        self.val_dataset = torch.load( os.path.join( dataset_dir, 'val_dataset.pt' ) )
        self.test_dataset = torch.load( os.path.join( dataset_dir, 'test_dataset.pt' ) )
        self.train_adj_mat = torch.load( os.path.join( dataset_dir, 'train_adj_mat.pt' ) )
        self.train_mask = torch.load( os.path.join( dataset_dir, 'train_mask.pt' ) )

        self.n_users = self.train_adj_mat.shape[0]
        self.n_items = self.train_adj_mat.shape[1]

        self.load_dataset( r )
        self.sampling()

    def __len__( self ):
        return self.pos_train.shape[0]

    def __getitem__( self, idx ):
        return self.pos_train[ idx ], self.neg_train[ idx ]

    def get_val( self ):
        return self.val_dataset

    def get_test( self ):
        return self.test_dataset

    def get_reg_mat( self ):
        return self.interact_mapper

    def load_dataset( self, r ):
        interact_mapper = torch.load( os.path.join( self.relation_dir, str(r), 'interact.pt' ) )

        self.interact_mapper = interact_mapper

    def sampling( self ):
        relation_adj_mat = self.train_adj_mat * self.train_mask
        relation_neg_adj_mat = ( 1 - self.train_adj_mat ) * self.train_mask

        user_mask = torch.sum( relation_adj_mat, dim=-1 ) > 0

        num_samples = torch.sum( relation_adj_mat, dim=-1 ).tolist()
        neg_train = torch.zeros( ( 0, 2 ), dtype=torch.long )

        for uid in torch.arange( self.train_adj_mat.shape[0] )[ user_mask ].tolist():
            neg_items = torch.multinomial( relation_neg_adj_mat[ uid ], num_samples=int( num_samples[ uid ] ), replacement=True ).reshape( -1, 1 )
            user_idx = torch.full( ( int( num_samples[ uid ] ), 1 ), uid, dtype=torch.long )
            neg_train = torch.vstack( ( neg_train, torch.hstack( ( user_idx, neg_items ) ) ) )

        self.pos_train, self.neg_train = relation_adj_mat.nonzero(), neg_train

if __name__ == '__main__':
    dataset = Ml1mDataset( 2 )
    print( dataset.get_reg_mat().shape )
    print( dataset[256] )

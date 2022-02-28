import os
import torch
from torch.utils.data import Dataset

class Ml1mDataset( Dataset ):
    def __init__( self, relation ):
        self.relation = relation
        self.dataset_dir = './process_datasets/ml-1m/'

        self.load_dataset()

        self.n_users, self.n_items = self.adj_mat.shape

    def __len__( self ):
        return self.n_users

    def __getitem__( self, idx ):
        return idx, self.train_adj_mat[ idx ], self.confidence_mat[ self.relation ][ idx ]

    def get_val( self ):
        return self.val_mask > 0, self.adj_mat * self.val_mask

    def get_test( self ):
        return self.test_mask > 0, self.adj_mat * self.test_mask

    def get_reg_mat( self ):
        return self.interact[ self.relation ].T

    def load_dataset( self ):
        self.adj_mat = torch.load( os.path.join( self.dataset_dir, 'adj_mat.pt' ) )
        self.train_mask = torch.load( os.path.join( self.dataset_dir, 'train_mask.pt' ) )
        self.val_mask = torch.load( os.path.join( self.dataset_dir, 'val_mask.pt' ) )
        self.test_mask = torch.load( os.path.join( self.dataset_dir, 'test_mask.pt' ) )
        self.interact = torch.load( os.path.join( self.dataset_dir, 'interact.pt' ) )
        self.confidence_mat = torch.load( os.path.join( self.dataset_dir, 'sim_relation_mat.pt' ) )

        self.train_adj_mat = self.adj_mat * self.train_mask

if __name__ == '__main__':
    dataset = Ml1mDataset( 'item_genre' )
    print( dataset[0] )

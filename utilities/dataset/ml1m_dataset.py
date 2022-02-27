import os
import torch
from torch.utils.data import Dataset

class Ml1mDataset( Dataset ):
    def __init__( self ):
        self.dataset_dir = './process_datasets/ml-1m/'

        self.load_dataset()

        self.n_users, self.n_items = self.adj_mat.shape
        self.sampling()

    def __len__( self ):
        return self.pos_train_data.shape[0]

    def __getitem__( self, idx ):
        return self.pos_train_data[ idx ], self.neg_train_data[ idx ] 

    def get_val( self ):
        return self.val_mask > 0, self.adj_mat * self.val_mask

    def get_test( self ):
        return self.test_mask > 0, self.adj_mat * self.test_mask

    def get_reg_mat( self, relation : str ):
        return self.interact[ relation ]

    def load_dataset( self ):
        self.adj_mat = torch.load( os.path.join( self.dataset_dir, 'adj_mat.pt' ) )
        self.train_mask = torch.load( os.path.join( self.dataset_dir, 'train_mask.pt' ) )
        self.val_mask = torch.load( os.path.join( self.dataset_dir, 'val_mask.pt' ) )
        self.test_mask = torch.load( os.path.join( self.dataset_dir, 'test_mask.pt' ) )
        self.interact = torch.load( os.path.join( self.dataset_dir, 'interact.pt' ) )

    def sampling( self ):
        pos_adj_mat = ( self.adj_mat * self.train_mask )
        neg_adj_mat = ( 1 - ( self.adj_mat * self.train_mask ) )

        num_neg_samples = torch.sum( pos_adj_mat, dim=-1 ).to( torch.int ).tolist()

        self.pos_train_data = pos_adj_mat.nonzero()
        self.neg_train_data = torch.zeros( ( 0, 2 ), dtype=torch.long )

        for user in range( self.n_users ):
            user_idx = torch.full( ( num_neg_samples[user], 1 ), user )
            neg_items = torch.multinomial( neg_adj_mat[user], num_samples=num_neg_samples[user] ).reshape( -1, 1 )
            self.neg_train_data = torch.vstack( ( self.neg_train_data, torch.hstack( ( user_idx, neg_items ) ) ) )

if __name__ == '__main__':
    dataset = Ml1mDataset()
    print( dataset[1999] )

import os
import torch
from torch.utils.data import Dataset

class Ml1mDataset( Dataset ):
    def __init__( self ):
        self.dataset_dir = './process_datasets/ml-1m/'

        self.load_dataset()

        self.n_users, self.n_items = self.adj_mat.shape
        self.sub_sampling()
        self.sampling()


    def __len__( self ):
        return self.pos_train_data.shape[0]

    def __getitem__( self, idx ):
        return self.pos_train_data[ idx ], self.neg_train_data[ idx ] 

    def get_val( self ):
        return self.val_mask, self.adj_mat * self.val_mask

    def get_test( self ):
        return self.test_mask, self.adj_mat * self.test_mask

    def get_reg_mat( self, relation : str ):
        return self.interact[ relation ]

    def load_dataset( self ):
        self.adj_mat = torch.load( os.path.join( self.dataset_dir, 'adj_mat.pt' ) )
        self.train_mask = torch.load( os.path.join( self.dataset_dir, 'train_mask.pt' ) )
        self.val_mask = torch.load( os.path.join( self.dataset_dir, 'val_mask.pt' ) )
        self.test_mask = torch.load( os.path.join( self.dataset_dir, 'test_mask.pt' ) )
        self.interact = torch.load( os.path.join( self.dataset_dir, 'interact.pt' ) )

    def sub_sampling( self ):
        pos_adj_mat = self.adj_mat * self.train_mask
        neg_adj_mat = 1 - ( self.adj_mat * self.train_mask )

        pos_item_prob = torch.sum( pos_adj_mat, dim=0 ) / self.n_users
        neg_item_prob = torch.sum( neg_adj_mat, dim=0 ) / self.n_users

        norm_pos_item_prob = ( torch.sqrt( pos_item_prob / 1e-3 ) + 1 ) * ( 1e-3 / pos_item_prob )
        norm_neg_item_prob = ( torch.sqrt( neg_item_prob / 1e-3 ) + 1 ) * ( 1e-3 / neg_item_prob )

        self.prob_pos_items = norm_pos_item_prob
        self.prob_neg_items = norm_neg_item_prob

    def sampling( self ):
        prob = torch.FloatTensor( self.n_items ).normal_(0, 1)
        pos_item_mask = self.prob_pos_items > prob
        neg_item_mask = self.prob_neg_items > prob

        pos_adj_mat = ( self.adj_mat * self.train_mask ) * pos_item_mask
        neg_adj_mat = ( 1 - ( self.adj_mat * self.train_mask ) ) * neg_item_mask

        user_mask = torch.sum( pos_adj_mat, dim=-1 ) > 0

        pos_items = torch.multinomial( pos_adj_mat[ user_mask ], num_samples=1 )
        neg_items = torch.multinomial( neg_adj_mat[ user_mask ], num_samples=1 )
        user_idx = torch.arange( self.n_users )[ user_mask ].reshape( -1, 1 )

        self.pos_train_data = torch.hstack( ( user_idx, pos_items ) )
        self.neg_train_data = torch.hstack( ( user_idx, neg_items ) )

if __name__ == '__main__':
    dataset = Ml1mDataset()

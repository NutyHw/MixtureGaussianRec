import os
import torch
from torch.utils.data import IterableDataset, DataLoader

class Ml1mDataset( IterableDataset ):
    def __init__( self, relation ):
        self.relation = relation
        self.dataset_dir = './process_datasets/ml-1m/'

        self.load_dataset()
        self.apply_mask()

        self.n_users, self.n_items = self.adj_mat.shape
        self.user_sim, self.item_sim = self.compute_confidence_mat()


    def __iter__( self ):
        return self.samples( 32 )

    def get_val( self ):
        return self.val_mask > 0, self.adj_mat * self.val_mask

    def get_test( self ):
        return self.test_mask > 0, self.adj_mat * self.test_mask

    def compute_metapath_sim( self, commute ):
        diag = torch.diag( commute ).reshape( 1, -1 ) + torch.diag( commute ).reshape( -1, 1 )
        return 2 * commute / diag

    def compute_confidence_mat( self ):
        user_commute = None
        item_commute = None

        if self.relation == 'item_genre':
            user_commute = torch.linalg.multi_dot( ( self.train_adj_mat, self.interact[ self.relation ].T, self.interact[ self.relation ], self.train_adj_mat.T ) )
            item_commute = torch.linalg.multi_dot( ( self.interact[ self.relation ].T, self.interact[ self.relation ] ) )


        elif self.relation == 'user_age' or self.relation == 'usre_jobs':
            item_commute = torch.linalg.multi_dot( ( self.train_adj_mat.T, self.interact[ self.relation ].T, self.interact[ self.relation ], self.train_adj_mat ) )
            user_commute = torch.linalg.multi_dot( ( self.interact[ self.relation ].T, self.interact[ self.relation ] ) )

        user_sim = self.compute_metapath_sim( user_commute )
        item_sim = self.compute_metapath_sim( item_commute )

        user_sim[ torch.arange( self.n_users ), torch.arange( self.n_users ) ]  = 0
        item_sim[ torch.arange( self.n_items ), torch.arange( self.n_items ) ]  = 0

        return user_sim, item_sim

    def apply_mask( self ):
        mask = torch.sum( self.interact[ 'item_genre' ].T, dim=-1 ) > 0
        self.train_adj_mat = self.train_adj_mat[ :, mask ] 
        self.adj_mat = self.adj_mat[ :, mask ]
        self.train_mask = self.train_mask[ :, mask ]
        self.val_mask = self.val_mask[ :, mask ]
        self.test_mask = self.test_mask[ :, mask ]
        self.interact[ 'item_genre' ] = self.interact['item_genre'][ :, mask ]

    def load_dataset( self ):
        self.adj_mat = torch.load( os.path.join( self.dataset_dir, 'adj_mat.pt' ) )
        self.train_mask = torch.load( os.path.join( self.dataset_dir, 'train_mask.pt' ) )
        self.val_mask = torch.load( os.path.join( self.dataset_dir, 'val_mask.pt' ) )
        self.test_mask = torch.load( os.path.join( self.dataset_dir, 'test_mask.pt' ) )
        self.interact = torch.load( os.path.join( self.dataset_dir, 'interact.pt' ) )

        self.train_adj_mat = self.adj_mat * self.train_mask

    def samples( self, batch_size ):
        shuffle_users = torch.randperm( self.n_users )
        for i in range( 0, self.n_users, batch_size ):
            end_idx = i + batch_size if i + batch_size < self.n_users else self.n_users
            sample_users = shuffle_users[ i : end_idx ]

            pos_sample_item = torch.flatten( torch.multinomial( self.train_adj_mat[ sample_users ], num_samples=1 ) )
            neg_sample_item = torch.flatten( torch.multinomial( 1 - self.train_adj_mat[ sample_users ], num_samples=1 ) )

            unique_items, inverse = torch.unique( torch.hstack( ( pos_sample_item, neg_sample_item ) ), return_inverse=True, sorted=True )

            comb_user = torch.combinations( sample_users, r=2 )
            comb_item = torch.combinations( unique_items, r=2 )

            user_sim = self.user_sim[ comb_user[:,0], comb_user[:,1] ]
            item_sim = self.item_sim[ comb_item[:,0], comb_item[:,1] ]

            yield sample_users, unique_items, user_sim, item_sim, inverse[ : sample_users.shape[0] ], inverse[ sample_users.shape[0] : ]

if __name__ == '__main__':
    dataset = Ml1mDataset( 'item_genre' )
    for i, batch in enumerate( dataset ):
        print( i, batch )


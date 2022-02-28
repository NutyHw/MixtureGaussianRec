import os
import torch
from torch.utils.data import Dataset

class Ml1mDataset( Dataset ):
    def __init__( self, relation ):
        self.relation = relation
        self.dataset_dir = './process_datasets/ml-1m/'

        self.load_dataset()

        self.n_users, self.n_items = self.adj_mat.shape
        self.compute_similarity_mat()
        self.save_process()

    def __len__( self ):
        return self.n_users

    def __getitem__( self, idx ):
        return idx, self.confidence_mat[ self.relation ][ idx ]

    def get_val( self ):
        return self.val_mask > 0, self.adj_mat * self.val_mask

    def get_test( self ):
        return self.test_mask > 0, self.adj_mat * self.test_mask

    def get_reg_mat( self ):
        return self.interact[ self.relation ]

    def load_dataset( self ):
        self.adj_mat = torch.load( os.path.join( self.dataset_dir, 'adj_mat.pt' ) )
        self.train_mask = torch.load( os.path.join( self.dataset_dir, 'train_mask.pt' ) )
        self.val_mask = torch.load( os.path.join( self.dataset_dir, 'val_mask.pt' ) )
        self.test_mask = torch.load( os.path.join( self.dataset_dir, 'test_mask.pt' ) )
        self.interact = torch.load( os.path.join( self.dataset_dir, 'interact.pt' ) )

    def compute_metapath_sim( self, metapath ):
        return  2 * metapath \
            / ( torch.diagonal( metapath ).reshape( -1, 1 ) + torch.diagonal( metapath ).reshape( 1, -1 ) )

    def compute_confidence( self, sim_mat, train_adj_mat ):
        confidence_mat = torch.zeros( ( 0, self.n_items ) )
        for user in range( self.n_users ):
            confidence_mat = torch.vstack(
                (
                    confidence_mat,
                    torch.mean(
                        sim_mat * train_adj_mat[ user ].reshape( 1, -1 ), dim=0
                    )
                )
            )
        return confidence_mat

    def compute_similarity_mat( self ):
        # compute item_category_category_item
        train_adj_mat = self.adj_mat * self.train_mask

        item_genre_sim = self.compute_metapath_sim(
            torch.chain_matmul( 
                self.interact['item_genre'].T, self.interact['item_genre']
            )
        )

        item_age_sim = self.compute_metapath_sim(
            torch.chain_matmul(
                train_adj_mat.T, self.interact['user_age'].T, self.interact['user_age'], train_adj_mat
            )
        )

        item_job_sim = self.compute_metapath_sim(
            torch.chain_matmul(
                train_adj_mat.T, self.interact['user_jobs'].T, self.interact['user_jobs'], train_adj_mat
            )
        )

        self.confidence_mat = {
            'item_genre' : self.compute_confidence( item_genre_sim, train_adj_mat ),
            'user_jobs' : self.compute_confidence( item_job_sim, train_adj_mat ),
            'user_age' : self.compute_confidence( item_age_sim, train_adj_mat )
        }

    def save_process( self ):
        torch.save( self.confidence_mat, 'sim_relation_mat' )


if __name__ == '__main__':
    dataset = Ml1mDataset( 'item_genre' )
    print( dataset[0] )

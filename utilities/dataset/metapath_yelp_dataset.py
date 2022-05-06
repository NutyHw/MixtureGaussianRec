import math
import sys
import os
import random
from collections import defaultdict
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class YelpDataset( Dataset ):
    def __init__( self, process_dataset, i, input_file, neg_size = 10 ):
        self.dataset = torch.load( os.path.join( process_dataset, f'yelp_dataset_{i}.pt' ) )

        with open( os.path.join( process_dataset, input_file ) , 'rb' ) as f:
            arr = np.load( f )

            user_cluster_mu = torch.from_numpy( arr['user_cluster_mu'] ).to( torch.float )
            user_cluster_cov = torch.from_numpy( arr['user_cluster_cov'] ).to( torch.float )
            item_cluster_mu = torch.from_numpy( arr['item_cluster_mu'] ).to( torch.float )
            item_cluster_cov = torch.from_numpy( arr['item_cluster_cov'] ).to( torch.float )

            gel_mat = self.compute_gaussian_expected_likehood( ( user_cluster_mu, user_cluster_cov ), ( item_cluster_mu, item_cluster_cov ) )
            self.norm_gel_mat = ( gel_mat - torch.mean( gel_mat, dim=-1 ).reshape( -1, 1 ) ) / torch.std( gel_mat, dim=-1 ).reshape( -1, 1 )

            self.user_weight = torch.from_numpy( arr['user_weight'] ).to( torch.float )
            self.item_weight = torch.from_numpy( arr['item_weight'] ).to( torch.float )

        self.n_users, self.n_items = self.dataset['train_adj'].shape
        self.true_prob = ( self.dataset[ 'train_adj' ] > 0 ).to( torch.float )
        self.true_prob = self.true_prob / torch.sum( self.true_prob, dim=-1 ).reshape( -1, 1 )

    def compute_kl_div( self, p : torch.Tensor, q : torch.Tensor):
        mu_p, sigma_p = p
        mu_q, sigma_q = q

        num_latent  = mu_p.shape[1]

        mu_p, sigma_p = mu_p.unsqueeze( dim=0 ), sigma_p.unsqueeze( dim=0 )
        mu_q, sigma_q = mu_q.unsqueeze( dim=1 ), sigma_q.unsqueeze( dim=1 )


        log_sigma = torch.log( 
            torch.prod( sigma_q, dim=-1 ) \
            / torch.prod( sigma_p, dim=-1 )
        )
        trace_sigma = torch.sum( ( 1 / sigma_q ) * sigma_p, dim=-1 )

        sum_mu_sigma = torch.sum( torch.square( mu_p - mu_q ) * ( 1 / sigma_q ), dim=-1 )

        return 0.5 * ( log_sigma + trace_sigma - num_latent + sum_mu_sigma ).T

    def compute_gaussian_expected_likehood( self, p : torch.Tensor, q : torch.Tensor ):
        mu_p, sigma_p = p
        mu_q, sigma_q = q

        num_latent = mu_p.shape[1]

        sigma_p, sigma_q = sigma_p.unsqueeze( dim=2 ), sigma_q.T.unsqueeze( dim=0 )
        mu_p, mu_q = mu_p.unsqueeze( dim=2 ), mu_q.T.unsqueeze( dim=0 )

        return torch.exp(
            0.5 * ( -\
                torch.log( torch.prod( sigma_p + sigma_q, dim=1 ) ) -\
                num_latent * math.log( 2 * math.pi ) -\
                torch.sum( ( mu_p - mu_q ) ** 2 * ( 1 / ( sigma_p + sigma_q ) ), dim=1 )
            )
        )

    def create_interact( self, neg_size : int ):
        train_adj_mat = self.dataset[ 'train_adj' ]
        pos_interact = ( train_adj_mat > 0 ).nonzero()
        neg_interact = torch.randint( self.n_items, ( pos_interact.shape[0], neg_size ) )

        self.pos_interact = pos_interact
        self.neg_interact = neg_interact

    def val_interact( self ):
        val_data = self.dataset['val_dataset']
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return val_data, y

    def test_interact( self ):
        val_data = self.dataset['test_dataset']
        y = torch.hstack( ( torch.ones( ( self.n_users, 1 ) ), torch.zeros( ( self.n_users, 100 ) ) ) )

        return val_data, y

    def __len__( self ):
        return self.pos_interact.shape[0]

    def __getitem__( self, idx ):
        return self.user_weight[ idx ], self.item_weight, self.norm_gel_mat, self.true_prob[ idx ]

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/', '1', 'yelp_gmm_result.npz', 20 )
    print( dataset[0] )
    #loader = DataLoader( dataset, batch_size=256 )
    #for idx, batch in enumerate( loader ):
    #    user_weight, item_weight, log_gauss_mat, true_prob = batch
    #    print( user_weight.shape, item_weight[0].shape ,log_gauss_mat[0].shape, true_prob.shape )

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ensemble_yelp_dataset import YelpDataset

class Encoder( nn.Module ):
    def __init__( self, L : list ):
        super().__init__()
        self.create_nn_structure( L )

    def create_nn_structure( self , L ):
        self.linear = list()
        for i in range( len( L ) - 2 ):
            self.linear.append( nn.Linear( L[i], L[i+1] ) )
            self.linear.append( nn.BatchNorm1d( L[i+1] ) )
            self.linear.append( nn.ReLU() )
            self.linear.append( nn.Dropout() )

        self.linear.append( nn.Linear( L[-2], L[-1] ) )
        self.linear = nn.ModuleList( self.linear )

    def forward( self, X ):
        for i, layer in enumerate( self.linear ):
            X = layer( X )
        return X

class GMF( nn.Module ):
    def __init__( self, num_latent ):
        super().__init__()
        self.model = nn.ModuleList( [
            nn.Linear( num_latent, 1, bias=False ),
        ] )

    def forward( self, X ):
        for i, layer in enumerate( self.model ):
            X = layer( X )
        return X

class GMMKlDiv( nn.Module ):
    def __init__( self ):
        super().__init__()

    def compute_kl_div( self, p : torch.Tensor, q : torch.Tensor):
        mu_p, sigma_p = torch.chunk( p, 2, dim=1 )
        mu_q, sigma_q = torch.chunk( q, 2, dim=1 )

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

    def compute_mixture_kl_div( self, k1, k2, kl_div_mat, kl_div_mat2 ):
        mixture1, mixture2 = k1, k2
        kl_div_mat = torch.exp( - kl_div_mat )
        kl_div_mat2 = torch.exp( - kl_div_mat2 )

        return torch.sum(
            mixture1.unsqueeze( dim=2 ) * torch.log( 
                torch.matmul( mixture1, kl_div_mat.T ).unsqueeze( dim=1 ) / torch.matmul( mixture2, kl_div_mat2.T ).unsqueeze( dim=0 )
            ).transpose( dim0=1, dim1=2 )
        ,dim=1 )

    def forward( self, k1, k2, p, q ):
        kl_div_mat = self.compute_kl_div( p, p )
        kl_div_mat2 = self.compute_kl_div( p, q )

        return self.compute_mixture_kl_div( k1, k2, kl_div_mat, kl_div_mat2 ), kl_div_mat2

class GmmExpectedKernel( nn.Module ):
    def __init__( self ):
        super().__init__()

    def compute_gaussian_expected_likehood( self, p : torch.Tensor, q : torch.Tensor ):
        mu_p, sigma_p = torch.chunk( p, 2, dim=1 )
        mu_q, sigma_q = torch.chunk( q, 2, dim=1 )
        print( mu_p.shape, sigma_p.shape )

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

    def compute_mixture_gaussian_expected_likehood( self, k1, k2, gaussian_mat ):
        return torch.log( torch.chain_matmul( k1, gaussian_mat, k2.T  ) )

    def forward( self, k1, k2, p, q ):
        gaussian_mat = self.compute_gaussian_expected_likehood( p, q )
        return self.compute_mixture_gaussian_expected_likehood( k1, k2, gaussian_mat ), torch.log( gaussian_mat )

class Ensemble( nn.Module ):
    def __init__( self, num_latent, gmf_state_dict, metapath_linear ):
        super().__init__()
        self.num_latent = num_latent

        # load and freeze gmf
        self.gmf = GMF( num_latent )
        self.gmf.load_state_dict( gmf_state_dict )
        self.gmf.requires_grad_( False )

        # load and freeze metapath
        self.metapath = nn.Linear( 3, 3 )
        self.metapath.load_state_dict( metapath_linear )
        self.metapath.requires_grad_( False )

        self.ensemble = nn.Linear( 2, 1 )

    def forward( self, user_embed, item_embed, user_weight, item_weight, log_gauss_mat ):
        temp = ( user_embed.unsqueeze( dim=1 ) * item_embed.unsqueeze( dim=0 ) ).reshape( -1, self.num_latent )
        embedding_prob = torch.softmax( self.gmf( temp ).reshape( -1, item_embed.shape[0] ), dim=-1 )
        metapath_prob = torch.softmax(
            torch.linalg.multi_dot( 
                ( user_weight, self.metapath( log_gauss_mat ), item_weight.T ) 
            ), dim=-1
        )

        ensemble_prob = self.ensemble( 
            torch.hstack( 
                ( embedding_prob.reshape( -1, 1 ), metapath_prob.reshape( -1, 1 ) ) 
            ) 
        ).reshape( user_embed.shape[0], item_embed.shape[0] )

        print( embedding_prob )
        print( metapath_prob )
        return ensemble_prob / torch.sum( ensemble_prob, dim=1 ).reshape( -1, 1 )

if __name__ == '__main__':
    dataset = YelpDataset( '../yelp_dataset/', 'cold_start' )
    loader = DataLoader( dataset, batch_size=32 )

    ensemble_state_dict = torch.load( 'ensemble_state_dict.pt' )
    ensemble = Ensemble( 64, ensemble_state_dict['gmf'], ensemble_state_dict['metapath'] )

    for idx, batch in enumerate( loader ):
        user_embed, item_embed, user_weight, item_weight, log_gauss_mat, true_prob = batch
        print( torch.max( ensemble( user_embed, item_embed[0], user_weight, item_weight[0], log_gauss_mat[0] ), axis=1 ) )
        break

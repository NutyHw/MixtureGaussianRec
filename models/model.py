import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Encoder( nn.Module ):
    def __init__( self, L : list ):
        super().__init__()
        self.create_nn_structure( L )

    def create_nn_structure( self , L ):
        self.linear = list()
        for i in range( len( L ) - 2 ):
            self.linear.append( nn.Linear( L[i], L[i+1] ) )
            self.linear.append( nn.ReLU() )
            self.linear.append( nn.Dropout() )

        self.linear.append( nn.Linear( L[-2], L[-1] ) )
        self.linear = nn.ModuleList( self.linear )

    def forward( self, X ):
        for i, layer in enumerate( self.linear ):
            X = layer( X )
        return X

class Decoder( nn.Module ):
    def __init__( self, L : list ):
        super().__init__()
        self.create_nn_structure( L )

    def create_nn_structure( self , L ):
        self.linear = list()
        for i in range( len( L ) - 2 ):
            self.linear.append( nn.Linear( L[i], L[i+1] ) )
            self.linear.append( nn.ReLU() )

        self.linear.append( nn.Linear( L[-2], L[-1] ) )
        self.linear = nn.ModuleList( self.linear )

    def forward( self, X ):
        for i, layer in enumerate( self.linear ):
            X = layer( X )
        return X

class CluserAssignment( nn.Module ):
    def __init__( self ):
        super().__init__()

    def memborship_assignment( self, X, cluster_mask ):
        '''
        argument 
        X : latent representation with shape ( N, num_latent )
        cluster_mask : mask matrix determine which cluser is sample belong to with shape ( N, num_cluster )

        return :
        probability of each sample belong to each cluster
        '''
        cluster_mu = torch.matmul( cluster_mask.T, X ) / torch.sum( cluster_mask, dim=0 ).reshape( -1, 1 )

        # studen t distribution kernel
        cluster_dist = torch.sum( torch.sqrt( torch.square( X.unsqueeze( dim=1 ) - cluster_mu.unsqueeze( dim=0 ) ) ), dim=-1 )
        q = ( 1 + cluster_dist ) ** -1
        norm_q = q / torch.sum( q, dim=-1 ).reshape( -1, 1 )

        return norm_q

    def forward( self, X : torch.Tensor, cluster_mask ):
        return self.memborship_assignment( X, cluster_mask )


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
        return self.compute_mixture_gaussian_expected_likehood( k1, k2, gaussian_mat ), gaussian_mat

#class ExpectedKernelModel( nn.Module ):
#    def __init__( self, n_users, n_items, user_mixture, item_mixture, num_latent, mean_constraint, sigma_min, sigma_max ):
#        super().__init__()
#        self.num_user_mixture  = user_mixture
#        self.num_item_mixture = item_mixture
#        self.user_gaussian = GaussianEmbedding( num_latent, user_mixture, mean_constraint, sigma_min, sigma_max )
#        self.item_gaussian = GaussianEmbedding( num_latent, item_mixture, mean_constraint, sigma_min, sigma_max )
#        self.user_mixture = MixtureEmbedding( user_mixture, n_users )
#        self.item_mixture = MixtureEmbedding( item_mixture, n_items )
#        self.expected_likehood_kernel = GmmExpectedKernel()
#
#    def forward( self, idx1, idx2, relation ):
#        mixture_1, mixture_2 = None, None
#
#        if relation == 'user-user':
#            mixture_1 =  self.user_mixture( idx1 )
#            mixture_2 = self.user_mixture( idx2 )
#            user_gaussian = self.user_gaussian( torch.arange( self.num_user_mixture ) )
#
#            return self.expected_likehood_kernel( mixture_1, mixture_2, user_gaussian, user_gaussian )
#
#        elif relation == 'user-item':
#            mixture_1 = self.user_mixture( idx1 )
#            mixture_2 = self.item_mixture( idx2 )
#
#            user_gaussian = self.user_gaussian( torch.arange( self.num_user_mixture ) )
#            item_gaussian = self.user_gaussian( torch.arange( self.num_item_mixture ) )
#           
#            return self.expected_likehood_kernel( mixture_1, mixture_2, user_gaussian, item_gaussian )
#
#        elif relation == 'item-item':
#            mixture_1 = self.item_mixture( idx1 )
#            mixture_2 = self.item_mixture( idx2 )
#
#            item_gaussian = self.user_gaussian( torch.arange( self.num_item_mixture ) )
#
#            return self.expected_likehood_kernel( mixture_1, mixture_2, item_gaussian, item_gaussian )

class ExpectedKernelModel( nn.Module ):
    def __init__( self, n_users, n_items, user_mixture, item_mixture, num_latent, beta ):
        super().__init__()
        self.beta = beta
        self.n_users = n_users
        self.n_items = n_items
        self.num_latent = num_latent
        self.num_user_mixture  = user_mixture
        self.num_item_mixture = item_mixture
        self.user_gaussian = GaussianEmbedding( num_latent, user_mixture )
        self.item_gaussian = GaussianEmbedding( num_latent, item_mixture )
        self.user_mixture = user_mixture
        self.item_mixture = item_mixture
        # self.user_mixture = MixtureEmbedding( user_mixture, n_users )
        # self.item_mixture = MixtureEmbedding( item_mixture, n_items )
        self.kl_div_kernel = GmmExpectedKernel()

    def regularization( self ):
        user_gaussian = self.user_gaussian()
        item_gaussian = self.item_gaussian()
        gaussian = torch.vstack( ( user_gaussian, item_gaussian ) )

        mu, sigma = torch.chunk( gaussian, 2, dim=1 )

        return 0.5 * torch.sum( torch.sum( mu ** 2, dim=1 ) + torch.sum( sigma, dim=1 ) - self.num_latent - torch.log( torch.prod( sigma, dim=1 ) ) ) 

    def mutual_distance( self ):
        user_gaussian = self.user_gaussian()
        item_gaussian = self.item_gaussian()

        user_kl_div = self.kl_div_kernel.compute_kl_div( user_gaussian, user_gaussian )
        item_kl_div = self.kl_div_kernel.compute_kl_div( item_gaussian, item_gaussian )

        return torch.sum( user_kl_div ), torch.sum( item_kl_div )

    def compute_transition_prob( self, user_group_prob, item_group_prob, kl_div_mat ):
        transition_prob = torch.softmax( - kl_div_mat * self.beta, dim=-1 )

        return torch.chain_matmul( user_group_prob, transition_prob, item_group_prob.T ) 

    def clustering_assignment_hardening( self, group_prob ):
        square_group_prob = group_prob ** 2
        temp = square_group_prob / torch.sum( group_prob, dim=0 ).reshape( 1, -1 )
        P = temp / torch.sum( temp, dim=1 ).reshape( -1, 1 )

        return F.kl_div( torch.log( P ), group_prob, reduction='sum' )

    def forward( self, user_idx, item_idx, is_test=False ):
        mixture_1 = self.user_mixture[ user_idx ]
        mixture_2 = self.item_mixture[ item_idx ]

        gaussian_1 = self.user_gaussian()
        gaussian_2 = self.item_gaussian()

        mixture_kl_div, kl_div_mat = self.kl_div_kernel( mixture_1, mixture_2, gaussian_1, gaussian_2 )

        if is_test:
            return mixture_kl_div
        else:
            return mixture_kl_div, self.compute_transition_prob( mixture_1, mixture_2, kl_div_mat ), mixture_1, mixture_2

#class DistanceKlDiv( nn.Module ):
#    def __init__( self, n_users, n_items, n_mixture, n_latent, attribute  ):
#        super().__init__()
#        assert( attribute in [ 'item_attribute', 'user_attribute' ] )
#        self.n_mixture  = n_mixture
#        self.n_users = n_users
#        self.n_items = n_items
#        self.attribute = attribute
#
#        self.global_gaussian = GaussianEmbedding( n_latent, n_mixture )
#        self.user_gaussian = GaussianEmbedding( n_latent, n_users )
#        self.item_gaussian = GaussianEmbedding( n_latent, n_items )
#
#    def compute_kl_div( self, p : torch.Tensor, q : torch.Tensor):
#        mu_p, sigma_p = torch.hsplit( p, 2 )
#        mu_q, sigma_q = torch.hsplit( q, 2 )
#
#        num_latent  = mu_p.shape[1]
#
#        mu_p, sigma_p = mu_p.unsqueeze( dim=0 ), sigma_p.unsqueeze( dim=0 )
#        mu_q, sigma_q = mu_q.unsqueeze( dim=1 ), sigma_q.unsqueeze( dim=1 )
#
#
#        log_sigma = torch.log( 
#            torch.prod( sigma_q, dim=-1 ) \
#            / torch.prod( sigma_p, dim=-1 )
#        )
#        trace_sigma = torch.sum( ( 1 / sigma_q ) * sigma_p, dim=-1 )
#
#        sum_mu_sigma = torch.sum( torch.square( mu_p - mu_q ) * ( 1 / sigma_q ), dim=-1 )
#
#        return 0.5 * ( log_sigma + trace_sigma - num_latent + sum_mu_sigma ).T
#
#    def regularization( self ):
#        all_gauss = torch.vstack( ( self.user_gaussian( torch.arange( self.n_users ) ), self.item_gaussian( torch.arange( self.n_items ) ) ) )
#        gauss_mu, gauss_sigma = torch.hsplit( all_gauss, 2 )
#        num_latent = gauss_mu.shape[1]
#
#        return 0.5 * (
#           torch.sum( torch.square( gauss_mu ), dim=-1 ) \
#           + torch.sum( gauss_sigma, dim=-1 ) \
#           - num_latent \
#           - torch.log( torch.prod( gauss_sigma, dim=-1 ) )
#       )
#
#    def forward( self, unique_user, unique_item ):
#        global_gauss = self.global_gaussian( torch.arange( self.n_mixture ) )
#        user_to_global = self.compute_kl_div( self.user_gaussian( unique_user ), global_gauss ) + self.compute_kl_div( global_gauss, self.user_gaussian( unique_user ) ).T
#        global_to_item = self.compute_kl_div( self.item_gaussian( unique_item ), global_gauss ).T + self.compute_kl_div( global_gauss, self.item_gaussian( unique_item ) )
#
#        user_item_dist =  user_to_global.unsqueeze( dim=2 ) + global_to_item.unsqueeze( dim=0 )
#
#        result = torch.min( user_item_dist, dim=1 )[0]
#
#        if self.attribute == 'user_attribute':
#            return result, unique_user, user_to_global
#
#        elif self.attribute == 'item_attribute':
#            return result, unique_item, global_to_item.T

if __name__ == '__main__':
   # dataset = Dataset( 'item_genre' )
   dataset = Dataset( './yelp_dataset/', 0, 'BCat', 40 )
   dataloader = DataLoader( dataset, batch_size=32 )
   layer = [ dataset.attribute.shape[1], 64 ] 
   encoder = Encoder( layer )
   decoder = Decoder( layer[::-1] )

   print( encoder )
   print( decoder )
   # print( summary( encoder, ( dataset.train_adj_mat.shape[1] ) ) )
   for i, batch in enumerate( dataloader ):
       adj = batch
       embed = encoder( adj )
       pred = decoder( embed )
       print( embed )
       print( pred )
       break


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import DataLoader
# from ml1m_dataset import Ml1mDataset as Dataset

# class GCN( MessagePassing ):
    # def __init__( self, in_dim, out_dim, n, m ):
        # super().__init__(aggr='mean')
        # self.linear =  nn.Linear( in_dim, out_dim, bias=True )
        # self.n, self.m = n, m

        # self.init_xavior()

    # def init_xavior( self ):
        # nn.init.xavier_uniform( self.linear.weight )

    # def forward( self, x, edge_index ):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=edge_index.size(0))
        # row, col = edge_index
        # # transform
        # x = self.linear( x )

        # # normalize
        # deg_n = degree(row, self.n, dtype=x.dtype )
        # deg_m = degree(col, self.m, dtype=x.dtype)
        # deg_inv_sqrt_n = deg_n.pow(-0.5)
        # deg_inv_sqrt_m = deg_m.pow(-0.5)
        # norm = deg_inv_sqrt_n[row] * deg_inv_sqrt_m[col]
        # deg_inv_sqrt_n[deg_inv_sqrt_n == float('inf')] = 0
        # deg_inv_sqrt_m[deg_inv_sqrt_m == float('inf')] = 0

        # return self.propagate( edge_index, x=x, norm=norm, size=( self.n, self.m ) )

    # def message( self, x_j, norm ):
        # return torch.tanh( norm.reshape( -1, 1 ) * x_j )

# class EmbeddingGCN( nn.Module ):
    # def __init__( self, num_latent, out_dim, num_hidden, n, m ):
        # super().__init__()
        # model = list()
        # assert( num_hidden % 2 == 0 )
        # for i in range( num_hidden ):
            # if i == num_hidden - 1:
                # model.append( GCN( num_latent, out_dim, m, n ) )
            # elif i % 2 == 0:
                # model.append( GCN( num_latent, num_latent, n, m ) )
            # else:
                # model.append( GCN( num_latent, num_latent, m, n ) )

        # self.model = nn.ModuleList( model )
        # self.embedding = nn.Embedding( num_embeddings=n, embedding_dim=num_latent )

        # nn.init.xavier_uniform( self.embedding.weight )

    # def forward( self, x, edge_index ):
        # x = self.embedding( x )

        # for i, model in enumerate( self.model ):
            # x = model( x, edge_index )
            # new_edge_index = torch.empty( edge_index.shape, dtype=torch.int64 ).detach()
            # new_edge_index[0] = edge_index[1]
            # new_edge_index[1] = edge_index[0]
            # edge_index = new_edge_index

        # return torch.softmax( x, dim=-1 )

class GaussianEmbedding( nn.Module ):
    def __init__( self, num_latent, n ):
        super().__init__()
        self.params = nn.ParameterDict({
            'mu' : nn.Parameter( torch.rand( ( n, num_latent ) ) ),
            'sigma' : nn.Parameter( torch.rand( ( n, num_latent ) ) )
        })
        self.init_xavior()

    def init_xavior( self ):
        nn.init.xavier_normal_( self.params['mu'] )
        nn.init.xavier_normal_( self.params['sigma'] )

    def forward( self ):
        return torch.hstack( ( self.params['mu'], torch.exp( self.params['sigma'] ) ) )

class MixtureEmbedding( nn.Module ):
    def __init__( self, num_mixture, n ):
        super().__init__()
        self.mixture = nn.Embedding( num_embeddings=n, embedding_dim=num_mixture )
        self.init_xavior()

    def init_xavior( self ):
        nn.init.xavier_normal_( self.mixture.weight )
    
    def forward( self, idx ):
        return torch.softmax( self.mixture( idx ), dim=-1 )

class GMMKlDiv( nn.Module ):
    def __init__( self ):
        super().__init__()

    def compute_kl_div( self, p : torch.Tensor, q : torch.Tensor):
        mu_p, sigma_p = torch.chunk( p, 2 )
        mu_q, sigma_q = torch.chunk( q, 2 )

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
        self.user_mixture = MixtureEmbedding( user_mixture, n_users )
        self.item_mixture = MixtureEmbedding( item_mixture, n_items )
        self.expected_kernel = GmmExpectedKernel()

    def mutual_distance( self ):
        user_gaussian = self.user_gaussian()
        item_gaussian = self.item_gaussian()

        user_mu = user_gaussian[ :, : self.num_latent ]
        item_mu = item_gaussian[ :, : self.num_latent ]

        user_distance = torch.sum( F.pdist( user_mu, 2 ) )
        item_distance = torch.sum( F.pdist( item_mu, 2 ) )

        return ( 2 * user_distance ) / self.num_user_mixture ** 2, ( 2 * item_distance ) / self.num_item_mixture ** 2

    def compute_transition_prob( self, user_group_prob, item_group_prob, kl_div_mat ):
        transition_prob = torch.softmax( torch.log( kl_div_mat ) * self.beta, dim=-1 )

        return torch.chain_matmul( user_group_prob, transition_prob, item_group_prob.T ) 

    def clustering_assignment_hardening( self, group_prob ):
        square_group_prob = group_prob ** 2
        temp = square_group_prob / torch.sum( group_prob, dim=0 ).reshape( 1, -1 )
        P = temp / torch.sum( temp, dim=1 ).reshape( -1, 1 )

        return F.kl_div( torch.log( P ), group_prob, reduction='sum' )

    def forward( self, user_idx, item_idx, is_test=False ):
        mixture_1 = self.user_mixture( user_idx )
        mixture_2 = self.item_mixture( item_idx )

        gaussian_1 = self.user_gaussian()
        gaussian_2 = self.item_gaussian()

        mixture_kl_div, kl_div_mat = self.expected_kernel( mixture_1, mixture_2, gaussian_1, gaussian_2 )

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
   model = ExpectedKernelModel( 600, 4000, 4, 4, 32, 1 )
   mixture_prob, transition_prob, user_group_prob, item_group_prob = model( torch.tensor([ 2, 3 ]).to( torch.long ), torch.arange( 20 ) ) 
   print( mixture_prob )
   print( transition_prob )


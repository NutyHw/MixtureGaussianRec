import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import transpose
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCN( MessagePassing ):
    def __init__( self, in_dim, out_dim, n, m ):
        super().__init__(aggr='mean')
        self.linear =  nn.Linear( in_dim, out_dim, bias=True )
        self.n, self.m = n, m

        self.init_xavior()

    def init_xavior( self ):
        nn.init.xavier_uniform( self.linear.weight )

    def forward( self, x, edge_index ):
        edge_index, _ = add_self_loops(edge_index, num_nodes=edge_index.size(0))
        row, col = edge_index
        # transform
        x = self.linear( x )

        # normalize
        deg_n = degree(row, self.n, dtype=x.dtype )
        deg_m = degree(col, self.m, dtype=x.dtype)
        deg_inv_sqrt_n = deg_n.pow(-0.5)
        deg_inv_sqrt_m = deg_m.pow(-0.5)
        norm = deg_inv_sqrt_n[row] * deg_inv_sqrt_m[col]
        deg_inv_sqrt_n[deg_inv_sqrt_n == float('inf')] = 0
        deg_inv_sqrt_m[deg_inv_sqrt_m == float('inf')] = 0

        return self.propagate( edge_index, x=x, norm=norm, size=( self.n, self.m ) )

    def message( self, x_j, norm ):
        return torch.tanh( norm.reshape( -1, 1 ) * x_j )

class EmbeddingGCN( nn.Module ):
    def __init__( self, num_latent, out_dim, num_hidden, n, m ):
        super().__init__()
        model = list()
        assert( num_hidden % 2 == 0 )
        for i in range( num_hidden ):
            if i == num_hidden - 1:
                model.append( GCN( num_latent, out_dim, m, n ) )
            elif i % 2 == 0:
                model.append( GCN( num_latent, num_latent, n, m ) )
            else:
                model.append( GCN( num_latent, num_latent, m, n ) )

        self.model = nn.ModuleList( model )
        self.embedding = nn.Embedding( num_embeddings=n, embedding_dim=num_latent )

        nn.init.xavier_uniform( self.embedding.weight )

    def forward( self, x, edge_index ):
        x = self.embedding( x )

        for i, model in enumerate( self.model ):
            x = model( x, edge_index )
            new_edge_index = torch.empty( edge_index.shape, dtype=torch.int64 ).detach()
            new_edge_index[0] = edge_index[1]
            new_edge_index[1] = edge_index[0]
            edge_index = new_edge_index

        return torch.softmax( x, dim=-1 )

class GaussianEmbedding( nn.Module ):
    def __init__( self, num_latent, n, mean_constraint, sigma_min, sigma_max ):
        super().__init__()
        self.mean_constraint = mean_constraint
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.mu = nn.Embedding( num_embeddings=n, embedding_dim=num_latent )
        self.sigma = nn.Embedding( num_embeddings=n, embedding_dim=num_latent )

    def constraint_distribution( self ):
        self.mu.weight.clamp_( - ( self.mean_constraint ** 0.5 ), self.mean_constraint ** 0.5 )
        self.sigma.weight.clamp_( math.log( self.sigma_min ), math.log( self.sigma_max ) )

    def init_xavior( self ):
        nn.init.xavier_uniform( self.mu.weight )

    def forward( self, idx : torch.LongTensor ):
        return torch.hstack( ( self.mu( idx ), F.elu( self.sigma( idx ) ) + 1 ) )

class MixtureEmbedding( nn.Module ):
    def __init__( self, num_mixture, n ):
        super().__init__()
        self.mixture = nn.Embedding( num_embeddings=n, embedding_dim=num_mixture )
    
    def forward( self, idx ):
        return torch.softmax( self.mixture( idx ), dim=-1 )

class GmmExpectedKernel( nn.Module ):
    def __init__( self ):
        super().__init__()

    def compute_gaussian_expected_likehood( self, p : torch.Tensor, q : torch.Tensor ):
        mu_p, sigma_p = torch.hsplit( p, 2 )
        mu_q, sigma_q = torch.hsplit( q, 2 )

        num_latent = mu_p.shape[1]

        sigma_p, sigma_q = sigma_p.unsqueeze( dim=2 ), sigma_q.T.unsqueeze( dim=0 )
        mu_p, mu_q = mu_p.unsqueeze( dim=2 ), mu_q.T.unsqueeze( dim=0 )

        return torch.exp(
            0.5 * ( -\
                torch.log( torch.prod( sigma_p + sigma_q, dim=1 ) ) -\
                num_latent * torch.log( torch.tensor( [ 2 * math.pi ] ) ) -\
                torch.sum( ( mu_p - mu_q ) ** 2 * ( 1 / ( sigma_p + sigma_q ) ), dim=1 )
            )
        )

    def compute_mixture_gaussian_expected_likehood( self, k1, k2, gaussian_mat ):
        return torch.log( torch.linalg.multi_dot( ( k1, gaussian_mat, k2.T ) ) )

    def forward( self, k1, k2, p, q ):
        gaussian_mat = self.compute_gaussian_expected_likehood( p, q )
        return self.compute_mixture_gaussian_expected_likehood( k1, k2, gaussian_mat )

class Model( nn.Module ):
    def __init__( self, n_users, n_items, n_mixture, num_latent, mean_constraint, sigma_min, sigma_max ):
        self.num_global = n_mixture
        self.global_gaussian = GaussianEmbedding( num_latent, n_mixture, mean_constraint, sigma_min, sigma_max )
        self.user_mixture = MixtureEmbedding( n_mixture, n_users )
        self.item_mixture = MixtureEmbedding( n_mixture, n_items )
        self.expected_likehood_kernel = GmmExpectedKernel()

    def forward( self, idx1, idx2, relation ):
        global_gaussian = self.global_gaussian( torch.arange( self.num_global ) )
        mixture_1, mixture_2 = None, None

        if relation == 'user-user':
           mixture_1 =  self.user_mixture( idx1 )
           mixture_2 = self.user_mixture( idx2 )

        elif relation == 'user-item':
           mixture_1 = self.user_mixture( idx1 )
           mixture_2 = self.item_mixture( idx2 )

        elif relation == 'item-item':
            mixture_1 = self.item_mixture( idx1 )
            mixture_2 = self.item_mixture( idx2 )

        return self.expected_likehood_kernel( mixture_1, mixture_2, global_gaussian, global_gaussian )

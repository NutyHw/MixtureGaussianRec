import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MF( nn.Module ):
    def __init__( self, n_users, n_items, n_latent ):
        super().__init__()
        self.user_embed = nn.Embedding( n_users, n_latent )
        self.item_embed = nn.Embedding( n_items, n_latent )
        nn.init.xavier_uniform_( self.user_embed.weight )
        nn.init.xavier_uniform_( self.item_embed.weight )

    def compute_mu_sigma( self, cluster_mask, embed ):
        # input shape
        # cluster_mask = n_cluster x n
        # embed = n x n_latent
        # return shape 
        # mu = n_cluster x n_latet
        # sigma = n_cluster x n_latent x n_latent
        mu = torch.mm( cluster_mask, embed ) / torch.sum( cluster_mask, dim=-1 ).reshape( -1, 1 )
        sigma = torch.sum( torch.square( embed.unsqueeze( 1 ) - mu.unsqueeze( 0 ) ), dim=0 ) / torch.sum( cluster_mask, dim=-1 ).reshape( -1, 1 )
        return mu, sigma

    def soft_cluster_assignment( self, mu, embed ):
        # return shape
        # norm_q = n x n_cluster
        q = torch.sum( ( 1 + ( embed.unsqueeze( dim=1 ) - mu.unsqueeze( dim=0 ) ) ** 2 ) ** -1, dim=-1 )
        norm_q = q / torch.sum( q, dim=-1 ).reshape( -1, 1 )
        return norm_q

    def compute_nll( self, cluster_mask, is_user_cluster ):
        embed = None
        if is_user_cluster:
            embed = self.user_embed.weight
        else:
            embed = self.item_embed.weight

        mu, _ = self.compute_mu_sigma( cluster_mask, embed )
        mixture = self.soft_cluster_assignment( mu, embed )

        return mixture

    def forward( self, user_idx, item_idx, is_test = False):
        if is_test:
            return F.cosine_similarity( self.user_embed.weight.unsqueeze( dim=1 ), self.item_embed.weight.unsqueeze( dim=0 ), dim=-1 )
        return F.cosine_similarity( self.user_embed( user_idx ), self.user_embed( item_idx ) ).reshape( -1, 1 )

if __name__ == '__main__':
    mf = MF( 10, 10, 10 )
    user_idx = torch.tensor( [ 1, 6 ], dtype=torch.long )
    item_idx = torch.tensor( [ 2, 3 ], dtype=torch.long )

    res = mf( user_idx, item_idx, is_test=True )
    print( res.shape )
    mask = ( torch.rand( ( 4, 10 ) ) > 0.5 ).to( torch.float )
    print( mf.compute_nll( mask, True ) )


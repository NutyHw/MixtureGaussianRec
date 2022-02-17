import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class Model( nn.Module ):
    def __init__( self, num_user : int, num_item : int, num_category : int, num_group : int, num_latent : int, mean_constraint : float, sigma_min : float, sigma_max : float ):
        super( Model, self ).__init__()
        self.num_latent = num_latent
        self.mean_constraint = mean_constraint
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = 1e-9

        self.embedding = nn.ParameterDict({
            'user_embedding' : nn.Parameter( torch.normal( 0, 1, ( num_user, num_group ) ) ),
            'item_embedding' : nn.Parameter( torch.normal( 0, 1, ( num_item, num_category ) ) ),
            'group_mu' : nn.Parameter( torch.normal( 0, 1,( num_group, num_latent ) ) ),
            'group_log_sigma' : nn.Parameter( torch.normal( 0, 1,( num_group, num_latent ) ) ),
            'category_mu' : nn.Parameter( torch.normal( 0, 1,( num_category, num_latent ) ) ),
            'category_log_sigma' : nn.Parameter( torch.normal( 0, 1,( num_category, num_latent ) ) ),
        })

        self.transition_weight = nn.Linear( num_category, num_category, bias=True )

    def prob_encoder( self ):
        return torch.hstack( ( self.embedding['group_mu'], torch.exp( self.embedding['group_log_sigma'] ) ) ),\
                torch.hstack( ( self.embedding['category_mu'], torch.exp( self.embedding['category_log_sigma'] ) ) )

    def compute_weight( self, kl_dist_mat : torch.Tensor ):
        kl_dist_mat = ( kl_dist_mat - torch.mean( kl_dist_mat, dim=-1 ).reshape( -1, 1 ) ) / torch.std( kl_dist_mat, dim=-1 ).reshape( -1, 1 )
        return torch.sigmoid( self.transition_weight( kl_dist_mat ) )

    def compute_mixture_gaussian_expected_likehood( self, k1, k2, p, q ):
        k1, k2 = k1.unsqueeze( dim=2 ), k2.unsqueeze( dim=1 )
        return torch.log(
                    torch.sum(
                        torch.sum(
                            k1 * k2 * self.compute_gaussian_expected_likehood( p, q ).unsqueeze( dim=0 ),
                            dim=-1
                        ),
                        dim=-1
                    )
                ).reshape( -1, 1 )

    def compute_gaussian_expected_likehood( self, p : torch.Tensor, q : torch.Tensor ):
        mu_p, sigma_p = torch.hsplit( p, 2 )
        mu_q, sigma_q = torch.hsplit( q, 2 )

        sigma_p, sigma_q = sigma_p.unsqueeze( dim=2 ), sigma_q.T.unsqueeze( dim=0 )
        mu_p, mu_q = mu_p.unsqueeze( dim=2 ), mu_q.T.unsqueeze( dim=0 )

        return torch.exp(
            0.5 * ( -\
                torch.log( torch.prod( sigma_p + sigma_q, dim=1 ) ) -\
                self.num_latent * torch.log( torch.tensor( [ 2 * math.pi ] ) ) -\
                torch.sum( ( mu_p - mu_q ) ** 2 * ( 1 / ( sigma_p + sigma_q ) ), dim=1 )
            )
        )

    def constraint_distribution( self ):
        self.embedding['group_mu'].data.clamp_( - ( self.mean_constraint ** 0.5 ), self.mean_constraint ** 0.5 )
        self.embedding['category_mu'].data.clamp_( - ( self.mean_constraint ** 0.5 ), self.mean_constraint ** 0.5 )

        self.embedding['group_log_sigma'].data.clamp_( math.log( self.sigma_min ), math.log( self.sigma_max ) )
        self.embedding['category_log_sigma'].data.clamp_( math.log( self.sigma_min ), math.log( self.sigma_max ) )

    def forward( self, user_idx, item_idx, is_test = False ):
        self.constraint_distribution()
        unique_user_idx, user_indices = torch.unique( user_idx, return_inverse=True, sorted=True )
        group_prob, category_prob = self.prob_encoder()

        if is_test:
            user_k = torch.softmax( self.embedding['user_embedding'][ user_idx ], dim=-1 )
            item_k = torch.softmax( self.embedding['item_embedding'], dim=-1 )
            transition = self.compute_weight( self.compute_gaussian_expected_likehood( group_prob, category_prob ) )

            return torch.linalg.multi_dot( ( user_k, transition, item_k.T ) )

        unique_item_idx, item_indices = torch.unique( item_idx, return_inverse=True, sorted=True )

        user_k = torch.softmax( self.embedding['user_embedding'][ unique_user_idx ], dim=-1 )
        item_k = torch.softmax( self.embedding['item_embedding'][ unique_item_idx ], dim=-1 )

        transition = self.compute_weight( self.compute_gaussian_expected_likehood( group_prob, category_prob ) )
        kl_div = self.compute_mixture_gaussian_expected_likehood( user_k[ user_indices ], item_k[ item_indices ], group_prob, category_prob )

        return {
            'out' : kl_div,
            'kg_prob' : torch.linalg.multi_dot( ( user_k, transition, item_k.T ) )[ user_indices, item_indices ].reshape( -1, 1 ),
            'category_kg' : torch.log( item_k + self.eps )
        }

if __name__ == '__main__':
    model = Model( 943, 1682, 18, 10, 8, 100, 0.01, 10 )
    # print( model( torch.tensor([ 0, 1 ]), torch.tensor([ 1, 2 ]) ) )
    print( model( torch.tensor([ 0, 1 ]), torch.tensor([ 1, 2 ]), is_test=True ) )
    # optimizer = optim.RMSprop( model.parameters(), lr=1e-2 )

    # bce_loss = nn.BCELoss()

    # for i in range( 200 ):
        # optimizer.zero_grad( set_to_none=True )
        # res = model( g, torch.tensor( [ [ 2 ], [ 4 ] ] ), torch.tensor( [ [ 0 ], [ 1 ] ] ) )
        # loss = bce_loss( res, torch.tensor( [ [ 1 ], [ 0 ] ] ).to( torch.float32 ) )
        # print( loss )
        # loss.backward()
        # optimizer.step()


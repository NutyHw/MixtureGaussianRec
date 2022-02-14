import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class Model( nn.Module ):
    def __init__( self, num_user : int, num_item : int, num_category : int, num_group : int, num_latent : int ):
        super( Model, self ).__init__()
        self.num_latent = num_latent

        self.embedding = nn.ParameterDict({
            'user_embedding' : nn.Parameter( torch.normal( 0, 1,( num_user, num_group ) ) ),
            'item_embedding' : nn.Parameter( torch.normal( 0, 1,( num_item, num_category ) ) ),
            'group_embedding' : nn.Parameter( torch.normal( 3, 1,( num_group, num_latent * 2 ) ) ),
            'category_embedding' : nn.Parameter( torch.normal( 3, 1,( num_category, num_latent * 2 ) ) )
        })

        self.transition_weight = nn.Linear( num_category, num_category, bias=True )

    def compute_weight( self, kl_dist_mat : torch.Tensor ):
        return F.sigmoid( self.transition_weight( kl_dist_mat ) )

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

    def prob_encoder( self ):
        group_mu, log_group_sigma = torch.hsplit( self.embedding['group_embedding'], sections=2 )
        category_mu, log_category_sigma = torch.hsplit( self.embedding['category_embedding'], sections=2 )

        return torch.hstack( ( group_mu, F.elu( log_group_sigma ) + 1 ) ), torch.hstack( ( category_mu, F.elu( log_category_sigma ) + 1 ) )

    def forward( self, user_idx, item_idx, is_test = False ):
        unique_user_idx, user_indices = torch.unique( user_idx, return_inverse=True, sorted=True )

        if is_test:
            if not hasattr( self, 'transition' ):
                group_prob, category_prob = self.prob_encoder()
                self.transition = self.compute_weight( self.compute_gaussian_expected_likehood( group_prob, category_prob ) )

            user_k = torch.softmax( self.embedding['user_embedding'][ unique_user_idx ], dim=-1 )
            item_k = torch.softmax( self.embedding['item_embedding'], dim=-1 )

            return torch.linalg.multi_dot( ( user_k, self.transition, item_k.T ) )

        unique_item_idx, item_indices = torch.unique( item_idx, return_inverse=True, sorted=True )

        user_k = torch.softmax( self.embedding['user_embedding'][ unique_user_idx ], dim=-1 )
        item_k = torch.softmax( self.embedding['item_embedding'][ unique_item_idx ], dim=-1 )

        group_prob, category_prob = self.prob_encoder()
        transition = self.compute_weight( self.compute_gaussian_expected_likehood( group_prob, category_prob ) )
        kl_div = self.compute_mixture_gaussian_expected_likehood( user_k[ user_indices,: ], item_k[ item_indices,: ], group_prob, category_prob )

        return {
            'out' : kl_div,
            'kg_prob' : torch.linalg.multi_dot( ( user_k, transition, item_k.T ) )[ user_indices, item_indices ].reshape( -1, 1 ),
            'distribution' : ( group_prob, category_prob ),
            'category_kg' : torch.log( item_k )
        }

if __name__ == '__main__':
    model = Model( 943, 1682, 18, 10, 8 )
    print( model( torch.tensor([ 0, 1 ]), torch.tensor([ 1, 2 ]) ) )
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


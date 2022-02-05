import math
import torch
import torch.nn as nn

class Model( nn.Module ):
    def __init__( self, num_user : int, num_item : int, num_category : int, num_group : int, num_latent : int ):
        super( Model, self ).__init__()
        self.num_latent = num_latent

        self.embedding = nn.ParameterDict({
            'user_embedding' : nn.Parameter( torch.normal( 0, 1,( num_user, num_group ) ) ),
            'item_embedding' : nn.Parameter( torch.normal( 0, 1,( num_item, num_category ) ) ),
            'group_embedding' : nn.Parameter( torch.normal( 0, 1,( num_group, num_latent * 2 ) ) ),
            'category_embedding' : nn.Parameter( torch.normal( 0, 1,( num_category, num_latent * 2 ) ) )
        })

    def compute_weight( self, kl_dist_mat : torch.Tensor ):
        return kl_dist_mat / torch.sum( kl_dist_mat, dim=-1 ).view( -1, 1 )

    def prob_encoder( self ):
        group_mu, group_sigma = self.embedding['group_embedding'][:,:self.num_latent], torch.exp( self.embedding['group_embedding'][:,self.num_latent:] )
        category_mu, category_sigma = self.embedding['category_embedding'][:,:self.num_latent], torch.exp( self.embedding['category_embedding'][:,self.num_latent:] )

        return torch.cat( ( group_mu, group_sigma ** 2 ), dim=-1 ), torch.cat( ( category_mu, category_sigma ** 2 ), dim=-1 )

    def compute_expected_likehood( self, p : torch.Tensor, q : torch.Tensor ):
        num_latent = self.num_latent
        p_reshape = p.unsqueeze( dim=2 )
        q_reshape = q.T.unsqueeze( dim=0 )
        temp = 0.5 * ( 
            - torch.log( torch.prod( p_reshape[:,num_latent:,:] + q_reshape[:,num_latent:,:], dim=1 ) ) -\
            torch.sum( ( p_reshape[:,:num_latent,:] - q_reshape[:,:num_latent,:] ) ** 2 / ( p_reshape[:,num_latent:,:] + q_reshape[:,num_latent:,:] ), dim=1 ) -\
            num_latent * torch.log( 2 * torch.tensor( [ math.pi ] ) )
        )
        return torch.exp( temp )

    def mixture_expected_likehood( self, k1 : torch.Tensor, k2 : torch.Tensor, p : torch.Tensor, q : torch.Tensor ):
        res = k1.unsqueeze( dim=2 ) * k2.unsqueeze( dim=1 )
        res2 = self.compute_expected_likehood( p, q ).unsqueeze( dim=0 )
        return torch.log( torch.sum( torch.sum( res * res2, dim=-1 ), dim=-1 ).view( -1, 1 ) )

    def forward( self, user_idx, item_idx, is_test = False ):
        unique_user_idx, user_indices = torch.unique( user_idx, return_inverse=True, sorted=True )
        unique_item_idx, item_indices = torch.unique( item_idx, return_inverse=True, sorted=True )

        if is_test:
            user_k = torch.softmax( self.embedding['user_embedding'][ unique_user_idx,: ], dim=-1 )
            item_k = torch.softmax( self.embedding['item_embedding'][ unique_item_idx,: ], dim=-1 )

            return torch.log( torch.linalg.multi_dot( ( user_k, self.transition, item_k.T ) ) )[ user_indices, item_indices ].view( -1, 1 )

        user_k = torch.softmax( self.embedding['user_embedding'][unique_user_idx,:], dim=-1 )
        item_k = torch.softmax( self.embedding['item_embedding'][unique_item_idx,:], dim=-1 )

        group_prob, category_prob = self.prob_encoder()

        self.transition = self.compute_weight( self.compute_expected_likehood( group_prob, category_prob ) )

        kl_div = self.mixture_expected_likehood( user_k[ user_indices,: ], item_k[ item_indices,: ], group_prob, category_prob )

        return {
            'out' : kl_div,
            'kg_prob' : torch.log( torch.linalg.multi_dot( ( user_k, self.transition, item_k.T ) )[ user_indices, item_indices ] ).view( -1, 1 ),
            'distribution' : ( group_prob, category_prob ),
            'category_kg' : torch.log_softmax( self.embedding['item_embedding'][ unique_item_idx, : ], dim=-1 )
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


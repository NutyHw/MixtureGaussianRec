import math
import torch
import torch.nn.functional as F
import torch.jit as jit
import torch.nn as nn

class Model( nn.Module ):
    def __init__( self, **kwargs ):
        super( Model, self ).__init__()
        self.num_latent = kwargs['num_latent']

        self.embedding = nn.ParameterDict({
            'user_embedding' : nn.Parameter( torch.normal( 0, 1, ( kwargs['num_user'], kwargs['num_category'] ) ) ),
            'item_embedding' : nn.Parameter( torch.normal( 0, 1, ( kwargs['num_item'], kwargs['num_category'] ) ) ),
            'category_mu' : nn.Parameter( torch.normal( 0, 1,( kwargs['num_category'], kwargs['num_latent'] ) ) ),
            'category_log_sigma' : nn.Parameter( torch.normal( 0, 1,( kwargs['num_category'], kwargs['num_latent'] ) ) ),
        })

        # init 
        self.xavier_init()

    def xavier_init( self ):
        for embedding in self.embedding.keys():
            nn.init.xavier_normal_( self.embedding[embedding] )

    def _get_prob( self ):
        return self.embedding['category_mu'], \
            torch.exp( self.embedding['category_log_sigma'] )

    def _compute_kl_div( self, **kwargs):
        gaussian1_mu = kwargs['gaussian1_mu'].unsqueeze( dim=0 )
        gaussian1_sigma = kwargs['gaussian1_sigma'].unsqueeze( dim=0 )
        gaussian2_mu = kwargs['gaussian2_mu'].unsqueeze( dim=1 )
        gaussian2_sigma = kwargs['gaussian2_sigma'].unsqueeze( dim=1 )

        log_sigma = torch.log( 
            torch.prod( gaussian2_sigma, dim=-1 ) \
            / torch.prod( gaussian1_sigma, dim=-1 )
        )
        trace_sigma = torch.sum( ( 1 / gaussian2_sigma ) * gaussian1_sigma, dim=-1 )

        sum_mu_sigma = torch.sum( torch.square( gaussian1_mu - gaussian2_mu ) * ( 1 / gaussian2_sigma ), dim=-1 )

        return 0.5 * ( log_sigma + trace_sigma - self.num_latent + sum_mu_sigma ).T

    def _compute_mixture_kl_div( self, **kwargs ):
        mixture1, mixture2 = kwargs['mixture1'], kwargs['mixture2']
        kl_div_mat = torch.exp( - kwargs['category_category_kl_div_mat'] )

        return torch.sum(
            mixture1.unsqueeze( dim=2 ) * torch.log( 
                torch.matmul( mixture1, kl_div_mat.T ).unsqueeze( dim=1 ) / torch.matmul( mixture2, kl_div_mat.T ).unsqueeze( dim=0 )
            ).transpose( dim0=1, dim1=2 )
        ,dim=1 )

<<<<<<< HEAD

    def _compute_transition_prob( self, **kwargs):
        kl_div_mat = kwargs['group_category_kl_div_mat']
        temp = kl_div_mat.shape

        # normalize
        kl_div_mat = ( kl_div_mat - torch.mean( kl_div_mat, dim=-1 ).reshape( -1, 1 ) ) \
            / torch.std( kl_div_mat, dim=-1 ).reshape( -1, 1 )

        transition_prob = torch.sigmoid( self.linear_model( kl_div_mat.reshape( -1, 1 ) ).reshape( temp[0], temp[1] ) )

        prob = torch.linalg.multi_dot(
            ( kwargs['mixture1'], transition_prob, kwargs['mixture2'].T )
        )

        return prob / torch.sum( prob, dim=-1 ).reshape( -1, 1 )

=======
>>>>>>> 546b7f9c46bded89cdca84909ea70b531b516c12
    def _kl_div_to_normal_gauss( self ):
        category_mu, category_sigma = self.embedding['category_mu'], torch.exp( self.embedding['category_log_sigma'] )

        return 0.5 * (
            torch.sum( torch.square( category_mu ), dim=-1 ) \
            + torch.sum( category_sigma, dim=-1 ) \
            - self.num_latent \
            - torch.log( torch.prod( category_sigma, dim=-1 ) )
        )

    def forward( self, user_idx ):
        user_embedding = torch.softmax( self.embedding['user_embedding'][ user_idx ], dim=-1 )
        item_embedding = torch.softmax( self.embedding['item_embedding'], dim=-1 )

        gaussian_mu, gaussian_sigma = self._get_prob()

        category_category_kl_div_mat = self._compute_kl_div( 
            gaussian1_mu=gaussian_mu,
            gaussian2_mu=gaussian_mu,
            gaussian1_sigma=gaussian_sigma,
            gaussian2_sigma=gaussian_sigma
        )

        mixture_user_item_kl_div = self._compute_mixture_kl_div( 
            mixture1=user_embedding,
            mixture2=item_embedding,
            category_category_kl_div_mat=category_category_kl_div_mat
        )

        return mixture_user_item_kl_div, self._kl_div_to_normal_gauss(), user_embedding, item_embedding

if __name__ == '__main__':
    model = Model( num_user=943, num_item=1682, num_category=500, num_group=25, num_latent=128, beta=10, attribute='user_attribute' ) 
    trace_model = jit.trace( model, torch.tensor([ 0, 1 ]) )
    mixture_kl_div , regularization, user_embedding, item_embedding = trace_model( torch.tensor([ 0, 1 ]) )
    print( mixture_kl_div )
    # optimizer = optim.RMSprop( model.parameters(), lr=1e-2 )

    # bce_loss = nn.BCELoss()

    # for i in range( 200 ):
        # optimizer.zero_grad( set_to_none=True )
        # res = model( g, torch.tensor( [ [ 2 ], [ 4 ] ] ), torch.tensor( [ [ 0 ], [ 1 ] ] ) )
        # loss = bce_loss( res, torch.tensor( [ [ 1 ], [ 0 ] ] ).to( torch.float32 ) )
        # print( loss )
        # loss.backward()
        # optimizer.step()


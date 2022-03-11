import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class Model( nn.Module ):
    def __init__( self, **kwargs ):
        super( Model, self ).__init__()
        self.num_latent = kwargs['num_latent']
        self.mean_constraint = kwargs['mean_constraint']
        self.sigma_min = kwargs['sigma_min']
        self.sigma_max = kwargs['sigma_max']

        self.embedding = nn.ParameterDict({
            'group_mu' : nn.Parameter( torch.normal( 0, 1,( kwargs['num_group'], kwargs['num_latent'] ) ) ),
            'group_log_sigma' : nn.Parameter( torch.normal( 0, 1,( kwargs['num_group'], kwargs['num_latent'] ) ) ),
            'category_mu' : nn.Parameter( torch.normal( 0, 1,( kwargs['num_category'], kwargs['num_latent'] ) ) ),
            'category_log_sigma' : nn.Parameter( torch.normal( 0, 1,( kwargs['num_category'], kwargs['num_latent'] ) ) ),
        })

        if kwargs['attribute'] == 'user_attribute':
            self.embedding['user_embedding'] = nn.Parameter( 
                F.one_hot( torch.arange( kwargs['num_user'] ) % kwargs['num_category'] ).to( torch.float )
            )
            self.embedding['item_embedding'] = nn.Parameter( 
                F.one_hot( torch.arange( kwargs['num_item'] ) % kwargs['num_group'] ).to( torch.float )
            )
        elif kwargs['attribute'] == 'item_attribute':
            self.embedding['user_embedding'] = nn.Parameter( 
                F.one_hot( torch.arange( kwargs['num_user'] ) % kwargs['num_group'] ).to( torch.float )
            )
            self.embedding['item_embedding'] = nn.Parameter( 
                F.one_hot( torch.arange( kwargs['num_item'] ) % kwargs['num_category'] ).to( torch.float )
            )

        self.transition_weight = nn.Linear( 1, 1, bias=True )

        self.embedding['user_embedding'].register_hook( lambda grad : F.relu( grad ) )
        self.embedding['item_embedding'].register_hook( lambda grad : F.relu( grad ) )
        self.attribute = kwargs['attribute']

        self.xavior_init()

    def xavior_init( self ):
        for embedding in self.embedding:
            if 'embedding' in embedding:
                continue
            nn.init.xavier_uniform_( self.embedding[embedding] )
        nn.init.xavier_uniform_( self.transition_weight.weight )

    def prob_encoder( self ):
        p = torch.hstack( ( self.embedding['group_mu'], torch.exp( self.embedding['group_log_sigma'] ) ) )
        q = torch.hstack( ( self.embedding['category_mu'], torch.exp( self.embedding['category_log_sigma'] ) ) )

        if self.attribute == 'item_attribute':
            return p, q
        elif self.attribute == 'user_attribute':
            return q, p

    def compute_weight( self, kl_dist_mat : torch.Tensor ):
        kl_dist_mat = ( kl_dist_mat - torch.mean( kl_dist_mat, dim=-1 ).reshape( -1, 1 ) ) / torch.std( kl_dist_mat, dim=-1 ).reshape( -1, 1 )
        return torch.sigmoid( self.transition_weight( kl_dist_mat.reshape( -1, 1 ) ) ).reshape( kl_dist_mat.shape )

    def compute_mixture_gaussian_expected_likehood( self, k1, k2, gaussian_expected_likehood ):
        return torch.log( torch.linalg.multi_dot( ( k1, gaussian_expected_likehood, k2.T ) ) )

    def compute_gaussian_expected_likehood( self, p : torch.Tensor, q : torch.Tensor ):
        mu_p, sigma_p = torch.hsplit( p, 2 )
        mu_q, sigma_q = torch.hsplit( q, 2 )

        sigma_p, sigma_q = sigma_p.unsqueeze( dim=2 ), sigma_q.T.unsqueeze( dim=0 )
        mu_p, mu_q = mu_p.unsqueeze( dim=2 ), mu_q.T.unsqueeze( dim=0 )

        return torch.exp(
            0.5 * ( 
                - torch.log( torch.prod( sigma_p + sigma_q, dim=1 ) ) \
                - self.num_latent * torch.log( torch.tensor( [ 2 * math.pi ] ) ) \
                - torch.sum( ( mu_p - mu_q ) ** 2 * ( 1 / ( sigma_p + sigma_q ) ), dim=1 )
            )
        )

    def constraint_distribution( self ):
        self.embedding['group_mu'].data.clamp_( - ( self.mean_constraint ** 0.5 ), self.mean_constraint ** 0.5 )
        self.embedding['category_mu'].data.clamp_( - ( self.mean_constraint ** 0.5 ), self.mean_constraint ** 0.5 )

        self.embedding['group_log_sigma'].data.clamp_( math.log( self.sigma_min ), math.log( self.sigma_max ) )
        self.embedding['category_log_sigma'].data.clamp_( math.log( self.sigma_min ), math.log( self.sigma_max ) )

    def forward( self, user_idx, item_idx ):
        self.constraint_distribution()

        group_prob, category_prob = self.prob_encoder()

        unique_user_idx, inverse_user_idx = torch.unique( user_idx, return_inverse=True, sorted=True )
        unique_item_idx, inverse_item_idx = torch.unique( item_idx, return_inverse=True, sorted=True )

        user_k = self.embedding['user_embedding'][ unique_user_idx ] \
            / torch.sum( self.embedding['user_embedding'][ unique_user_idx ], dim=-1 ).reshape( -1, 1 )
        item_k = self.embedding['item_embedding'][ unique_item_idx ]  \
            / torch.sum( self.embedding['item_embedding'][ unique_item_idx ], dim=-1 ).reshape( -1, 1 )

        gaussian_expected_likehood = self.compute_gaussian_expected_likehood( group_prob, category_prob )
        transition = self.compute_weight( gaussian_expected_likehood )

        kl_div = self.compute_mixture_gaussian_expected_likehood(
            user_k,
            item_k,
            gaussian_expected_likehood
        )[ inverse_user_idx, inverse_item_idx ].reshape( -1, 1 )

        return kl_div, torch.linalg.multi_dot( ( user_k, transition, item_k.T ) )[ inverse_user_idx, inverse_item_idx ].reshape( -1, 1 ), user_k, item_k

if __name__ == '__main__':
    # print( model( torch.tensor([ 0, 1 ]), torch.tensor([ 1, 2 ]) ) )
    model = Model(
        num_latent=64,
        num_user=6040,
        num_item=3050,
        num_group=5,
        num_category=10,
        mean_constraint=10,
        sigma_min=0.1,
        sigma_max=5,
        attribute='user_attribute'
    )
    res = model( torch.tensor([ 0, 1 ] ), torch.tensor([ 1, 5 ]) )
    print( res[0].shape, res[1].shape, res[2].shape )
    # optimizer = optim.RMSprop( model.parameters(), lr=1e-2 )

    # bce_loss = nn.BCELoss()

    # for i in range( 200 ):
        # optimizer.zero_grad( set_to_none=True )
        # res = model( g, torch.tensor( [ [ 2 ], [ 4 ] ] ), torch.tensor( [ [ 0 ], [ 1 ] ] ) )
        # loss = bce_loss( res, torch.tensor( [ [ 1 ], [ 0 ] ] ).to( torch.float32 ) )
        # print( loss )
        # loss.backward()
        # optimizer.step()


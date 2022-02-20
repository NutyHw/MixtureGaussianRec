import os
import json
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

class EnsembleModel( nn.Module ):
    def __init__( self, n_users : int, model1 : nn.Module, model2 : nn.Module):
        super( EnsembleModel, self ).__init__()
        self.model1 = model1
        self.model2 = model2

        self.preference = nn.ParameterDict({
            'transition_preference' : nn.Parameter( torch.full( ( n_users, 2 ), 0.5 ) ),
            'prob_preference' : nn.Parameter( torch.full( ( n_users, 2 ), 0.5 ) ),
        })

    def forward( self, user_idx, item_idx, is_test = False ):
        unique_user_idx, user_indices = torch.unique( user_idx, return_inverse=True, sorted=True )
        res = self.model1( user_idx, item_idx, is_test=is_test )
        res2 = self.model2( user_idx, item_idx, is_test=is_test )

        if is_test:
            return res * self.preference['transition_preference'][ user_idx ][:,0].reshape( -1, 1 ) +\
                res2 * self.preference['transition_preference'][ user_idx ][:,1].reshape( -1, 1 )


        prob_preference = self.preference['prob_preference'][ user_idx ] / torch.sum( self.preference['prob_preference'][ user_idx ] ).reshape( -1, 1 )
        transition_preference = self.preference['transition_preference'][ user_idx ] / torch.sum( self.preference['transition_preference'][ user_idx ] ).reshape( -1, 1 )

        prob_res = torch.sum( 
            torch.hstack( ( res['out'], res2['out'] ) ) * prob_preference, dim=-1 
        ).reshape( -1, 1 )

        transition_res = torch.sum(
            torch.hstack( ( res['kg_prob'], res2['kg_prob'] ) ) * transition_preference, dim=-1 
        ).reshape( -1, 1 )

        return {
            'out' : prob_preference,
            'kg_prob' : transition_preference
        }

if __name__ == '__main__':
    # model = Model( 943, 1682, 18, 10, 8, 100, 0.01, 10 )
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


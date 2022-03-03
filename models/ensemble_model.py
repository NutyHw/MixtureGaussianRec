import os
import json
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from model import Model
import torch.jit as jit

class EnsembleModel( nn.Module ):
    def __init__( self, **kwargs ):
        super( EnsembleModel, self ).__init__()
        self.n_user = kwargs['num_user']
        self.n_item = kwargs['num_item']
        self.based_model = nn.ModuleList( kwargs['based_model'] )

        for based_model in self.based_model:
            for param in based_model.parameters():
                param.requires_grad = False

        self.preference = nn.ParameterDict({
            'transition_preference' : nn.Parameter( torch.full( ( self.n_user, len( kwargs['based_model'] ) ), 1.0 ) ),
            'prob_preference' : nn.Parameter( torch.full( ( self.n_user, len( kwargs['based_model'] ) ), 1.0 ) )
        })

    def _normalize_preference( self, user_idx ):
        return torch.softmax( self.preference[ 'prob_preference' ][ user_idx ], dim=-1 ).unsqueeze( dim=1 ), \
            torch.softmax( self.preference['transition_preference'][ user_idx ], dim=-1 ).unsqueeze( dim=1 )

    def forward( self, user_idx ):
        based_mixture_kl_div = list()
        based_transition_prob = list()

        for based_model in self.based_model:
            mixture_kl_div, transition_prob, _, _, _ = based_model( user_idx )
            based_mixture_kl_div.append( mixture_kl_div )
            based_transition_prob.append( transition_prob )

        mixture_preference, transition_preference = self._normalize_preference( user_idx )

        based_mixture_kl_div = torch.dstack( based_mixture_kl_div )
        based_transition_prob = torch.dstack( based_transition_prob )

        mixture_prob = torch.sum( based_mixture_kl_div * mixture_preference, dim=-1 )
        transition_prob = torch.sum( based_transition_prob * transition_preference, dim=-1 )

        return mixture_prob, transition_prob

if __name__ == '__main__':
    model1 = Model( num_user=943, num_item=1682, num_category=500, num_group=25, num_latent=128, beta=10, attribute='user_attribute' ) 
    model2 = Model( num_user=943, num_item=1682, num_category=500, num_group=25, num_latent=128, beta=10, attribute='user_attribute' ) 
    ensemble_model = EnsembleModel( num_user=943, num_item=1682, based_model=[ model1, model2 ] )
    mixture_kl_div, transition_prob = ensemble_model( torch.tensor([ 0, 1 ]) )
    print( mixture_kl_div.shape, transition_prob.shape )

import torch
import numpy as np

class SiameseDataset( object ):
    def __init__( self, affinity_file ):
        with open( affinity_file, 'rb' ) as f:
            self.affinity = torch.from_numpy( np.load( f ) )
            self.affinity -= torch.eye( self.affinity.shape[0] )

        self.prob_keep_word = self.compute_keeping_prob()

    def compute_keeping_prob( self ):
        z = torch.sum( self.affinity, dim=0 ) / self.affinity.shape[0]
        return ( ( z / 1e-3 ) ** 0.5 + 1 ) * ( 1e-3 / z )

    def neg_sampling( self, size ):
        freq = torch.sum( self.affinity, dim=0 )
        neg_prob = ( freq ** 0.75 ) / torch.sum( freq ** 0.75 )
        return torch.from_numpy( np.random.choice( self.affinity.shape[0], size, p=neg_prob ) )

    def samples( self, neg_samples ):
        prob = torch.rand( ( self.affinity.shape[0],  ) )
        mask = prob < self.prob_keep_word

        sampling_space = self.affinity[ mask ][ :, mask ]

        anchor, pos_anchor = sampling_space.nonzero( as_tuple=True )

        user_space = mask.nonzero().flatten()

        return user_space[ anchor ].tile( neg_samples ), user_space[ pos_anchor ].tile( neg_samples ), self.neg_sampling( anchor.shape[0] * neg_samples )


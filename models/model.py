import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseModel( nn.Module ):
    def __init__( self, n, num_latent, num_hidden ):
        super().__init__()
        model = list()
        for i in range( num_hidden ):
            if i == 0:
                model.append( nn.Linear( n, num_latent ) )
            else:
                model.append( nn.Linear( num_latent, num_latent ) )
            model.append( nn.LeakyReLU() )

        self.model = nn.ModuleList( model )
        self.xavier_init()

    def xavier_init( self ):
        for i in range( len (self.model) ):
            if i % 2 == 0:
                nn.init.xavier_uniform_( self.model[i].weight )

    def forward( self, x ):
        for i in range( len( self.model ) ):
            x = self.model[i]( x )
        return x

if __name__  == '__main__':
    x = torch.rand( ( 10, 100 ) )
    model = SiameseModel( 100, 64, 8 )
    print( model( x ).shape )


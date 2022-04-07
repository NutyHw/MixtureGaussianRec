import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.utils.data import DataLoader
from torch_cluster import random_walk

class GCN( nn.Module ):
    def __init__( self, in_dim, num_latent, num_hidden, activation='relu' ):
        super().__init__()
        model = list()
        for i in range( num_hidden ):
            if i == 0:
                model.append( SAGEConv( in_dim, num_latent,  normalize=True ) )
            else:
                model.append( SAGEConv( num_latent, num_latent, normalize=True ) )

            if activation == 'relu':
                model.append( nn.ReLU() )
            elif activation == 'tanh':
                model.append( nn.Tanh() )

            if i < num_hidden - 1:
                model.append( nn.Dropout( p=0.2 ) )

        self.model = nn.ModuleList( model )

    def forward( self, x, edge_indices ):
        for i in range( len( self.model ) ):
            if i % 3 == 0:
                x = self.model[i]( x, edge_indices )
            else:
                x = self.model[i]( x )
        return x

if __name__ == '__main__':
    edge_indices = ( torch.rand( ( 30, 30 ) ) - torch.eye( 30 ) > 0.5 ).nonzero().T
    model = GCN( 10, 64, 64, activation='relu' )
    x = torch.rand( ( 30, 10 ) )
    print( model( x, edge_indices ) )

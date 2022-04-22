import sys
import os
import random
from collections import defaultdict
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class YelpDataset( Dataset ):
    def __init__( self, process_dataset, i, attribute, alpha ):
        self.dataset = torch.load( os.path.join( process_dataset, f'yelp_dataset_{i}.pt' ) )

        with open( os.path.join( process_dataset, 'attribute.npy' ), 'rb' ) as f:
            arr = np.load( f )
            if attribute[0] == 'U':
                self.is_user_attribute = True
            elif attribute[0] == 'B':
                self.is_user_attribute = False

            if attribute == 'UU':
                self.attribute = torch.from_numpy( arr['UU'] )
            elif attribute == 'UCom':
                self.attribute = torch.from_numpy( arr['UCom'] )
            elif attribute == 'BCat':
                self.attribute = torch.from_numpy( arr['BCat'] )
            elif attribute == 'BCity':
                self.attribute = torch.from_numpy( arr['BCity'] )
            else:
                raise Exception('attribute invalid')

            self.attribute = F.normalize( self.attribute )

        self.create_input( alpha )

    def create_input( self, alpha ):
        train_adj = self.dataset['train_adj']
        confidence = ( 1 + alpha * train_adj ) * ( train_adj > 0 )
        norm_confidence = F.normalize( confidence )

        if self.is_user_attribute:
            temp = F.normalize( torch.mm( norm_confidence.T, self.attribute ) )
            self.attribute = torch.vstack( ( self.attribute, temp ) )
        else:
            temp = F.normalize( torch.mm( norm_confidence, self.attribute ) )
            self.attribute = torch.vstack( ( temp, self.attribute ) )

        self.attribute = self.attribute.to( torch.float )

    def __len__( self ):
        return self.attribute.shape[0]

    def __getitem__( self, idx ):
        return self.attribute[ idx ]

if __name__ == '__main__':
    dataset = YelpDataset( './yelp_dataset/', 0, 'BCat', 40 )
    print( dataset[0].type() )

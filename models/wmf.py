import torch
import torch.nn as nn

class WMF(nn.Module):
    def __init__(self, user_size : int, item_size : int, dim : int ):
        super().__init__()
        self.W = nn.Parameter(torch.normal( 0, 1, ( user_size, dim ) ))
        self.H = nn.Parameter(torch.normal( 0, 1, ( item_size, dim ) ) )

    def forward(self, user_idx : torch.Tensor, item_idx : torch.Tensor, is_test=False ):
        if is_test:
            return torch.matmul( self.W, self.H.T )

        user_embed = self.W[ user_idx ]
        item_embed = self.H[ item_idx ]

        y = torch.sum( user_embed * item_embed, dim=-1 )

        return y.reshape( -1, 1 )

if __name__ == '__main__':
    model = WMF( 10, 10, 16 )

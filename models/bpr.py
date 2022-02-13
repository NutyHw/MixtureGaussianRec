import torch
import torch.nn as nn

class BPR(nn.Module):
    def __init__(self, user_size : int, item_size : int, dim : int ):
        super().__init__()
        self.W = nn.Parameter(torch.normal( 0, 1, ( user_size, dim ) ))
        self.H = nn.Parameter(torch.normal( 0, 1, ( item_size, dim ) ) )

    def forward(self, user_idx : torch.Tensor, item_idx : torch.Tensor ):
        user_embed = self.W[user_idx, :]
        item_embed = self.H[item_idx, :]

        y = torch.mul(user_embed, item_embed).sum(dim=1).reshape( -1, 1 )

        return y

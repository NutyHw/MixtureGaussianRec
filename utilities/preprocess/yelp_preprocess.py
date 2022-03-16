from collections import defaultdict
import os
import torch

def load_file( fdir : str ):
    dataset = torch.load( os.path.join( fdir, 'dataset.pt' ) )
    metapath = torch.load( os.path.join( fdir, 'metapath.pt' ) )

    return dataset, metapath

def compute_jacard_sim( attribute_tensor : torch.Tensor ):
    '''
    attribute_tensor shape ( n, m )
    m => number of attribute, n => number of point
    '''

    all_jacard_sim = torch.zeros( ( attribute_tensor.shape[0], attribute_tensor.shape[0] ) )
    non_zero_indices = attribute_tensor.nonzero()

    temp = defaultdict( lambda : set() )
    for i in range( non_zero_indices.shape[0] ):
        temp[ non_zero_indices[i][0].item() ].add( non_zero_indices[i][1].item() )

    for i in range( attribute_tensor.shape[0] - 1 ):
        for j in range( i + 1, attribute_tensor.shape[0] ):
            intersect = len( temp[i].intersection( temp[j] ) )
            union = len( temp[i].union( temp[j] ) )
            jacard_sim = 0
            if union > 0:
                jacard_sim = intersect / union
            all_jacard_sim[ i, j ] = jacard_sim
            all_jacard_sim[ j, i ] = jacard_sim

    return all_jacard_sim

if __name__ == '__main__':
    dataset, metapaths = load_file( '../../process_datasets/yelp/' )

    metapath_jacard_sim = dict()

    for metapath in metapaths.keys():
        print(f'start {metapath}')
        if metapath == 'UU':
            continue
        jacard_sim = compute_jacard_sim( metapaths[ metapath ] )
        metapath_jacard_sim[ metapath ] = jacard_sim

    print( f'start interaction')
    interaction_jacard_sim = {
        'user_user' : compute_jacard_sim( dataset['train_adj_mat'] ),
        'item_item' : compute_jacard_sim( dataset['train_adj_mat'].T )
    }

    torch.save( metapath_jacard_sim, 'yelp_metapath_jacard_sim.pt' )
    torch.save( interaction_jacard_sim, 'yelp_interaction_jacard_sim.pt' )



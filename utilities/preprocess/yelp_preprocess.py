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

    jacard_sim = torch.zeros( ( attribute_tensor.shape[0], attribute_tensor.shape[0] ) )
    for i in range( attribute_tensor.shape[0] ):
        intersect = torch.sum( attribute_tensor[i].unsqueeze( dim=0 ) * attribute_tensor[ i + 1 : ], dim=-1 )
        union = torch.sum( ( attribute_tensor[i].unsqueeze( dim=0 ) + attribute_tensor[ i + 1 : ] ) > 0, dim=-1 )
        jacard_sim[ i, i + 1 : ] = intersect / union

    return jacard_sim

if __name__ == '__main__':
    dataset, metapaths = load_file( '../../process_datasets/yelp/' )

    metapath_jacard_sim = dict()

    for metapath in metapaths.keys():
        print( f'start metapath : { metapath }' )
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



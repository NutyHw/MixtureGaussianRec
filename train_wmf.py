import os
import json
from functools import partial
import torch
import implicit
import ray
from ray import tune
from scipy.sparse import csr_matrix
from ndcg import ndcg

from utilities.dataset.bpr_dataset import YelpDataset as Dataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

def construct_confidence_mat( dataset ):
    train_interact = ( dataset.dataset[ 'train_adj' ] > 0 ).to( torch.float ).nonzero()
    row = train_interact[:,0]
    col = train_interact[:,1]
    data = torch.ones( row.shape )
    return csr_matrix( 
        ( data, ( row, col ) ), shape=( dataset.n_users, dataset.n_items ) 
    )

def evaluate( true_rating, predict_rating, hr_k, recall_k, ndcg_k ):
    user_mask = torch.sum( true_rating, dim=-1 ) > 0
    predict_rating = predict_rating[ user_mask ]
    true_rating = true_rating[ user_mask ]

    _, top_k_indices = torch.topk( predict_rating, k=hr_k, dim=1, largest=True )
    hr_score = torch.mean( ( torch.sum( torch.gather( true_rating, dim=1, index=top_k_indices ), dim=-1 ) > 0 ).to( torch.float ) )

    _, top_k_indices = torch.topk( predict_rating, k=recall_k, dim=1, largest=True )

    recall_score = torch.mean( 
        torch.sum( torch.gather( true_rating, dim=1, index = top_k_indices ), dim=1 ) /
        torch.minimum( torch.sum( true_rating, dim=1 ), torch.tensor( [ recall_k ] ) )
    )

    ndcg_score = torch.mean( ndcg( predict_rating, true_rating, [ ndcg_k ] ) )

    return hr_score.item(), recall_score.item(), ndcg_score.item()

def validate_model( model, dataset, csr_matrix ):
    user_id = torch.arange( dataset.n_users )
    item_id, score = model.recommend( user_id, csr_matrix, N=dataset.n_items )

    val_interact, val_test_y = dataset.val_interact()
    test_interact, _ = dataset.test_interact()

    y_pred = torch.zeros( ( dataset.n_users, dataset.n_items ) ).scatter_( 1, torch.tensor( item_id ).to( torch.int64 ), torch.tensor( score ) )
    val_y_pred = torch.gather( y_pred, 1, val_interact )
    val_hr_score, val_recall_score, val_ndcg_score = evaluate( val_test_y, val_y_pred, 1, 10, 10)

    test_y_pred = torch.gather( y_pred, 1, test_interact )
    test_hr_score, test_recall_score, test_ndcg_score = evaluate( val_test_y, val_y_pred, 1, 10, 10)

    tune.report({
        'val_hr_score' : val_hr_score,
        'val_recall_score' : val_recall_score,
        'val_ndcg_score' : val_ndcg_score,
        'test_hr_score' : test_hr_score,
        'test_recall_score' : test_recall_score,
        'test_ndcg_score' : test_ndcg_score
    })

def train_model( config : dict, dataset ):
    dataset = ray.get( dataset )
    csr_matrix = construct_confidence_mat( dataset )

    model = implicit.als.AlternatingLeastSquares( 
        factors=config['num_latent'],
        regularization=config['gamma'],
        iterations=10
    )

    model.fit( csr_matrix, show_progress=True )

    validate_model( model, dataset, csr_matrix )

if __name__ == '__main__':
    ray.init( num_cpus=8 )
    dataset = ray.put( Dataset( './yelp_dataset/', '1' ) )
    config = {
        'num_latent' : 64,
        'gamma' : tune.grid_search([ 1e-4, 1e-3, 1e-2, 1e-1 ]),
    }

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 2 },
        config=config,
        metric='_metric/val_ndcg_score',
        mode='max',
        num_samples=1,
        local_dir='.',
        name='yelp_wmf_1'
    )

    with open('best_model.json','w') as f:
        json.dump( { 
            'best_model' : str( analysis.best_trial )
        }, f )

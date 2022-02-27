import os
import json
from functools import partial
import torch
import implicit
import ray
from ray import tune
from scipy.sparse import csr_matrix
from ndcg import ndcg

from utilities.dataset.ml1m_dataset import Ml1mDataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def construct_confidence_mat( dataset ):
    train_interact = dataset.pos_train_data
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

    y_pred = torch.zeros( ( dataset.n_users, dataset.n_items ) ).scatter_( 1, torch.tensor( item_id ).to( torch.int64 ), torch.tensor( score ) )

    mask, true_y = dataset.get_val()
    y_pred[ ~mask ] = 0

    val_hr_score, val_recall_score, val_ndcg_score = evaluate( true_y, y_pred, 1, 10, 10)

    mask, true_y = dataset.get_test()
    y_pred = torch.zeros( ( dataset.n_users, dataset.n_items ) ).scatter_( 1, torch.tensor( item_id ).to( torch.int64 ), torch.tensor( score ) )
    y_pred[ ~mask ] = 0
    test_hr_score, test_recall_score, test_ndcg_score = evaluate( true_y, y_pred, 1, 10, 10)

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
        iterations=1
    )

    model.fit( csr_matrix, show_progress=True )

    validate_model( model, dataset, csr_matrix )

if __name__ == '__main__':
    dataset = ray.put( Ml1mDataset() )
    config = {
        'num_latent' : tune.randint( 10, 200 ),
        'gamma' : tune.uniform( 1e-5, 1e-1 ),
    }

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        config=config,
        metric='_metric/val_ndcg_score',
        mode='max',
        num_samples=1,
        local_dir='/data2/saito/',
        name='train_wmf'
    )

    with open('best_model.json','w') as f:
        json.dump( { 
            'best_model' : str( analysis.best_trial )
        }, f )

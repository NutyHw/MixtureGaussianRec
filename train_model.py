import os
from functools import partial
import torch
import torch.nn as nn
from models.model import Model
import ray
import numpy as np
from ray import tune
from torch.utils.data import TensorDataset, DataLoader
from utilities.dataset.pantip_dataset import PantipDataset as Dataset
from torch.optim import Adagrad
from ndcg import ndcg
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def joint_loss( pos_result1, neg_result1, pos_result2, neg_result2 ):
    return torch.sum( torch.relu( - ( pos_result1 - neg_result1 ) * ( pos_result2 - neg_result2 ) ) )

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

def train_model( config, dataset=None, checkpoint_dir=None ):
    config = config
    dataset = ray.get( dataset )
    n_users, n_items = dataset.n_users, self.dataset.n_items
    reg_mat = dataset.get_reg_mat( config['relation'] )
    reg_mat  = ( reg_mat + 1e-6 ) / torch.sum( reg_mat, dim=-1 ).reshape( -1, 1 )

    config['num_user'] = n_users
    config['num_item'] = n_items
    config['num_category'] = reg_mat.shape[1]
    config['num_group'] = int( round( config['num_group'] ) )

    model = Model( **config )

    prediction_loss = nn.MarginRankingLoss( margin=config['prediction_margin'], reduction='mean' )
    transition_loss = nn.MarginRankingLoss( margin=config['transition_margin'], reduction='mean' )
    kl_div_loss = nn.KLDivLoss( reduction='batchmean' )

    alpha = config['alpha']
    beta = config['beta']
    gamma = config['gamma']

    loader = dataset.train_dataloader()
    val_loader = DataLoader( TensorDataset( torch.arange( n_users ).reshape( -1, 1 ) ), batch_size=32, shuffle=False, num_workers=1 )

    optimizer = Adagrad( params=model.parameters(), lr=config['lr'] )

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range( 10 ):
        for i, interact in enumerate( loader ):
            pos_interact, neg_interact = interact
            batch_size = pos_interact.shape[0]
            counter = 0
            for j in range( 0, batch_size, 2048 ):
                counter += 1
                end_idx = i + 2048

                if end_idx > batch_size:
                    end_idx = batch_size

                sub_batch_size = end_idx - i

                input_idx = torch.cat( ( pos_interact[i:end_idx], neg_interact[i:end_idx] ), dim=0 )
                res = model( input_idx[:,0], input_idx[:,1] - n_users )

                pos_res_out, neg_res_out = torch.split( res[0], split_size_or_sections=sub_batch_size, dim=0 )
                pos_res_kg_prob, neg_res_kg_prob = torch.split( res[1], split_size_or_sections=sub_batch_size, dim=0 )

                # prediction loss
                l1_loss = prediction_loss( pos_res_out, neg_res_out, torch.ones( ( batch_size, 1 ) ) )
                l2_loss = transition_loss( pos_res_kg_prob, neg_res_kg_prob, torch.ones( ( batch_size, 1 ) ) )
                l3_loss = joint_loss( pos_res_out, neg_res_out, pos_res_kg_prob, neg_res_kg_prob )

                # regularization loss
                item_idx = torch.unique( input_idx[:,1] - n_users, sorted=True )
                category_reg = kl_div_loss( torch.log( reg_mat[ item_idx ] ), res[3] )

                optimizer.zero_grad( set_to_none=True )
                loss = l1_loss * alpha + l2_loss * beta + l3_loss + gamma * category_reg
                
                tune.report( { 'loss' : loss } )
                loss.backward()
                optimizer.step()

        result = torch.zeros( ( 0, n_items ) )
        for idx, batch in enumerate( val_loader ):
            res, _, _, _ = model( batch[0][:,0], torch.arange( n_items ), is_test=True  )
            result = torch.vstack( ( result, res ) )

        val_mask, val_score = dataset.get_val()
        result[ ~val_mask ] = - np.inf
        hr, recall, ndcg = evaluate( val_score, result, 20, 100, 500 )

        tune.report( {
            'hr' : hr.item(),
            'recall' : recall.item(),
            'ndcg' : ndcg.item()
        } )

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

    
if __name__ == '__main__':
    ray.init( num_cpus=8,  _temp_dir='/data2/saito/' )
    dataset = ray.put( Dataset( './process_datasets/pantip_dataset/user_interact_window_1_window_2.pt', batch_size=32 ) )

    config = {
        # parameter to find
        'num_latent' : 64,

        # hopefully will find right parameter
        'prediction_margin' : tune.uniform( 1, 5 ),
        'transition_margin' : tune.uniform( 0.01, 0.5 ),
        'num_group' : tune.randint(4,20),
        'gamma' : tune.uniform( 1e-5, 1e-1 ),
        'lr' : 1e-3,
        'alpha' : 1,
        'beta' : 1,
        'mean_constraint' : 10,
        'sigma_min' : 0.1,
        'sigma_max' : 10,

        # fix parameter
        'relation' : 'rooms',
    }


    reporter = CLIReporter( [ 'hr', 'recall', 'ndcg' ], [ 'num_group', 'gamma' ] )
    scheduler = ASHAScheduler(
        metric='recall',
        mode='max',
        max_t=256,
        grace_period=1,
        reduction_factor=2
    )

    analysis = tune.run( 
        partial( train_model, dataset=dataset ),
        resources_per_trial={ 'cpu' : 1 },
        progress_reporter=reporter,
        num_samples=100,
        verbose=1,
        config=config,
        name=f'pantip_dataset_rooms',
        keep_checkpoints_num=2,
        local_dir=f"/data2/saito/",
        checkpoint_score_attr='ndcg_score',
    )



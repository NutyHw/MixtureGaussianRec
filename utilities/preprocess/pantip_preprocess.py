import  os
import pandas as pd
import json
from datetime import  datetime
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
import numpy as np
import torch

def connect() -> MongoClient:
    client = MongoClient(
        'orion.mikelab.net',
        username='saito',
        password='3ad-mCf-74p-gEE',
        authSource='pantip_recommend',
        authMechanism='SCRAM-SHA-1'
    )
    return client.pantip_recommend

def split_dataset():
    start = datetime( 2020, 9, 1 )
    end = datetime( 2021, 4, 1 )
    db = connect()

    counter = 0
    while start < end:
        print( start )
        pipeline = [
            { '$match' : { 'timestamp' : { '$gte' : start, '$lt' : start + relativedelta( months=1 ) } } },
            { '$out' : f'clickstream_window_{counter}' }
        ]
        db['dataset_daily_0.7_freq_1'].aggregate( pipeline )
        counter += 1
        start += relativedelta( months=1 )

def filter_items():
    pipeline = [
        { '$group' : { '_id' : '$item_id', 'count' : { '$sum' : 1 } } },
        { '$match' : { 'count' : { '$lt' : 10 } } }
    ]

    db = connect()
    for i in range( 7 ):
        print(f'preprocess {i}')
        cursor = db[f'clickstream_window_{i}'].aggregate( pipeline )

        filtered_item = list()
        for record in cursor:
            filtered_item.append( record['_id'] )

        db[f'clickstream_window_{i}'].delete_many({ 'item_id' : { '$in' : filtered_item } })

def create_dataset():
    db = connect()

    for i in range( 6 ):
        print(f'finish create dataset {i}')
        train_collection = f'clickstream_window_{i}'
        test_val_collection = f'clickstream_window_{i+1}'

        item_id = set( db[train_collection].distinct( 'item_id' ) )
        item_id2 = set( db[test_val_collection].distinct( 'item_id' ) )

        user_id = set( db[train_collection].distinct( 'user_id' )  )
        user_id2 = set( db[test_val_collection].distinct( 'user_id' )  )

        intersect_items = list( item_id.intersection( item_id2 ) )
        intersect_users = list( user_id.intersection( user_id2 ) )

        cursor = db[ train_collection ].find(
            {
                'user_id' : { '$in' : intersect_users },
                'item_id' : { '$in' : intersect_items }
            }
        )

        train_interact = list()

        for record in cursor:
            train_interact.append( [ intersect_users.index( record['user_id'] ), intersect_items.index( record['item_id'] ) ] )

        cursor = db[ test_val_collection ].find(
            {
                'user_id' : { '$in' : intersect_users },
                'item_id' : { '$in' : intersect_items }
            }
        )

        test_val_interact = list()

        for record in cursor:
            test_val_interact.append( [ intersect_users.index( record['user_id'] ), intersect_items.index( record['item_id'] ) ] )

        dataset_dir = f'dataset_window_{i}'
        os.mkdir( dataset_dir )

        torch.save( torch.tensor( train_interact ), os.path.join( dataset_dir, 'train_interact.pt' ) )
        torch.save( torch.tensor( test_val_interact ), os.path.join( dataset_dir, 'test_val_interact.pt' ) )
        torch.save( torch.tensor( [ int(user) for user in intersect_users ] ), os.path.join( dataset_dir, 'all_users.pt' ) )
        torch.save( torch.tensor( intersect_items ), os.path.join( dataset_dir, 'all_items.pt' ) )

if __name__ == '__main__':
    create_dataset()

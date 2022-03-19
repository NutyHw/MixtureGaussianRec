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

def filtered_no_tags_items():
    pipeline = [
        { '$group' : { '_id' : '$item_id' } },
        { '$lookup' : { 
            'from' : 'kratoos',
            'localField' : '_id',
            'foreignField' : 'topic_id',
            'as' : 'kratoo_data'
        }},
        { '$unwind' : '$kratoo_data' },
        { '$match' :  { '$or' : [ { 'kratoo_data.tags.0' : { '$exists' : False } }, { 'kratoo_data.room.0' : { '$exists' : False } } ] } }
    ]

    db = connect()
    for i in range( 7 ):
        print(f'preprocess {i}')
        cursor = db[f'clickstream_window_{i}'].aggregate( pipeline )
        filtered_item = list()
        for record in cursor:
            filtered_item.append( record['_id'] )
        filtered_item = list( set( filtered_item ) )
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

        all_rooms = list()
        all_tags = list()

        rooms_interact = list()
        tags_interact = list()

        cursor = db[ 'kratoos' ].find( { 'topic_id' : { '$in' : intersect_items } } )

        for record in cursor:
            for room in record['room']:
                if room in all_rooms:
                    rooms_interact.append( [ intersect_items.index( record['topic_id'] ), all_rooms.index( room ) ] )
                else:
                    rooms_interact.append( [ intersect_items.index( record['topic_id'] ), len( all_rooms ) ] )
                    all_rooms.append( room )
            for tag in record['tags']:
                if tag in all_tags:
                    tags_interact.append( [ intersect_items.index( record['topic_id'] ), all_tags.index( tag ) ] )
                else:
                    tags_interact.append( [ intersect_items.index( record['topic_id'] ), len( all_tags ) ] )
                    all_tags.append( tag )

        dataset_dir = f'dataset_window_{i}'
        os.mkdir( dataset_dir )

        torch.save( torch.tensor( train_interact ), os.path.join( dataset_dir, 'train_interact.pt' ) )
        torch.save( torch.tensor( test_val_interact ), os.path.join( dataset_dir, 'test_val_interact.pt' ) )
        torch.save( torch.tensor( [ int(user) for user in intersect_users ] ), os.path.join( dataset_dir, 'all_users.pt' ) )
        torch.save( torch.tensor( intersect_items ), os.path.join( dataset_dir, 'all_items.pt' ) )

        with open( os.path.join( dataset_dir, 'metadata_mapper.json' ), 'w' ) as f:
            json.dump( { 'rooms' : all_rooms, 'tags' : all_tags }, f )

        torch.save( torch.tensor( rooms_interact ), os.path.join( dataset_dir, 'item_rooms.pt' ) )
        torch.save( torch.tensor( tags_interact ), os.path.join( dataset_dir, 'item_tags.pt' ) )

if __name__ == '__main__':
    split_dataset()
    filtered_no_tags_items()
    create_dataset()

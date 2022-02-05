import sys
import os
import json
import re
from collections import defaultdict
import torch

class YelpPreprocess():
    def __init__( self, dataset_dir : str, save_dataset_dir : str ):
        self.train_file = os.path.join( dataset_dir, 'train1.txt' )
        self.test_file = os.path.join( dataset_dir, 'test.txt' )
        self.val_file = os.path.join( dataset_dir, 'valid1.txt' )
        self.users_file = os.path.join( dataset_dir, 'user_list.txt' )
        self.items_file = os.path.join( dataset_dir, 'item_list.txt' )
        self.kg_file = os.path.join( dataset_dir, 'kg_final.txt' )

        self.load_stats()

        train_cf_mat, val_cf_mat, test_cf_mat = self.load_cf( self.train_file ), self.load_cf( self.val_file ), self.load_cf( self.test_file )
        entity_mapper, interact_mapper = self.load_kg( self.kg_file )
        self.entity_mapper, self.interact_mapper = self.filter_reltion( entity_mapper, interact_mapper, 0.8 )

        self.val_dataset, self.test_dataset, self.train_adj_mat, self.train_mask = self.train_test_val_split( train_cf_mat, val_cf_mat, test_cf_mat, self.interact_mapper, self.entity_mapper )

        self.metapath_dataset = self.train_test_val_split_metapath( self.train_adj_mat, self.train_mask, self.interact_mapper )

        self.save_process_data( save_dataset_dir )

    def save_process_data( self, process_dir : str ):
        if os.path.exists( process_dir ):
            print(f'please remove { process_dir }')
            sys.exit()

        os.mkdir( process_dir )
        os.chdir( process_dir )

        torch.save( self.val_dataset, 'val_dataset.pt' )
        torch.save( self.test_dataset, 'test_dataset.pt' )
        torch.save( self.train_adj_mat, 'train_adj_mat.pt' )
        torch.save( self.train_mask, 'train_mask.pt' )

        os.mkdir( os.path.join( process_dir, 'relation_mat' ) )
        os.chdir( os.path.join( process_dir, 'relation_mat' ) )

        for r in self.metapath_dataset.keys():
            os.mkdir( str(r) )
            torch.save( self.metapath_dataset[ r ], os.path.join( str(r), f'relation_dataset.pt' ) )
            torch.save( self.entity_mapper[ r ], os.path.join( str(r), f'entity_mapper.pt' ) )
            torch.save( self.interact_mapper[ r ], os.path.join( str(r), f'interact_mapper.pt' ) )

    def load_stats( self ):
        with open( self.users_file ) as f:
            self.n_users = len( f.read().splitlines() )

        with open( self.items_file ) as f:
            self.n_items = len( f.read().splitlines() )

        with open( self.kg_file ) as f:
            self.n_entities = len( f.read().splitlines() )

    def load_cf( self, fname : str ):
        '''
        create interaction matrix where first column is user_id and second column is item_id
        '''
        users = list()
        items = list()
        with open( fname, 'r' ) as f:
            for line in f:
                row = [ int(i) for i in line.strip().split() ]
                if len( row ) > 1:
                    users += [ row[0] for i in range( len( row ) - 1 ) ]
                    items += row[1:]

        interact = torch.hstack( 
            (
                torch.tensor( users, dtype=torch.long ).reshape( -1, 1 ),
                torch.tensor( items, dtype=torch.long ).reshape( -1, 1 )
            )
        )

        return interact

    def load_kg( self, fname : str ):
        entity_mapper = defaultdict( lambda : [  ] )
        interact_mapper = defaultdict( lambda : torch.zeros( ( 0, 2 ), dtype=torch.long ) )

        with open( fname, 'r' ) as f:
            for line in f:
                u, r, v = line.strip().split()
                u, r, v = int(u), int(r), int(v)

                if u > self.n_items and v < self.n_items:
                    if u not in entity_mapper[ r ]:
                        entity_mapper[ r ].append( u )

                    interact = torch.tensor( [ [ entity_mapper[ r ].index( u ), v ] ], dtype=torch.long )
                    interact_mapper[ r ] = torch.vstack( ( interact_mapper[ r ], interact ) )

                elif v > self.n_items and u < self.n_items:
                    if v not in entity_mapper[ r ]:
                        entity_mapper[ r ].append( v )

                    interact = torch.tensor( [ [ entity_mapper[ r ].index( v ) , u ] ], dtype=torch.long )
                    interact_mapper[ r ] = torch.vstack( ( interact_mapper[ r ], interact ) )

        return entity_mapper, interact_mapper

    def filter_reltion( self, entity_mapper : defaultdict, interact_mapper : defaultdict, item_coverage_ratio : float ):
        # filter relations which have only 1 attribute or attribute with less than 10 items

        remove_relation = list()
        for r in entity_mapper.keys():
            # check if there are any attribute with less than 10 items

            entity_item_mat = torch.zeros( ( len( entity_mapper[r] ), self.n_items ) )
            entity_item_mat[ interact_mapper[r][:,0], interact_mapper[r][:,1] ] = 1

            num_item_per_attributes = torch.sum( entity_item_mat, dim=-1 )
            mask = num_item_per_attributes >= 10

            new_entity_mat = entity_item_mat[ mask, : ]

            entity_mapper[r] = torch.tensor( entity_mapper[r] )[ mask ]
            interact_mapper[r] = new_entity_mat

            num_relation_items = torch.sum( new_entity_mat, dim=0 ).count_nonzero()

            if num_relation_items < self.n_items * item_coverage_ratio:
                remove_relation.append( r )
                continue

            if entity_mapper[r].shape[0] == 1:
                remove_relation.append( r )

        for r in remove_relation:
            entity_mapper.pop( r )
            interact_mapper.pop( r )

        return entity_mapper, interact_mapper

    def create_mask( self, interact_mapper : defaultdict ):
        mask = dict()

        for r in interact_mapper.keys():
            mask[r] = torch.sum( interact_mapper[r], dim=0 )

        return mask

    def leave_one_out( self, interaction_adj_mat : torch.Tensor, mask : torch.Tensor ):
        user_mask = torch.sum( interaction_adj_mat * mask, dim=-1 ) >= 5

        val_test_items = torch.multinomial( interaction_adj_mat[ user_mask ] * mask[ user_mask ], num_samples=2 )
        neg_val_test_items = torch.multinomial( ( 1 - interaction_adj_mat[ user_mask ] ) * mask[ user_mask ], num_samples=200 )

        pos_val_items, pos_test_items = torch.hsplit( val_test_items, sections=2 )
        neg_val_items, neg_test_items = torch.hsplit( neg_val_test_items, sections=2 )

        val_items = torch.hstack( ( pos_val_items, neg_val_items ) ).reshape( -1, 1 )
        test_items = torch.hstack( ( pos_test_items, neg_test_items ) ).reshape( -1, 1 )

        user_ids = torch.arange( self.n_users )[ user_mask ].reshape( -1, 1 ).tile( 1, 101 ).reshape( -1, 1 )

        return torch.hstack( ( user_ids, val_items ) ), torch.hstack( ( user_ids, test_items ) )

    def train_test_val_split( self, train_cf_mat : torch.Tensor, val_cf_mat : torch.Tensor, test_cf_mat : torch.Tensor, interact_mapper : defaultdict, entity_mapper : defaultdict ):
        relation_mask = self.create_mask( interact_mapper )

        # create adj mat
        adj_mat = torch.zeros( ( self.n_users, self.n_items ) )
        adj_mat[ train_cf_mat[:,0], train_cf_mat[:,1] ] = 1
        adj_mat[ val_cf_mat[:,0], val_cf_mat[:,1] ] = 1
        adj_mat[ test_cf_mat[:,0], test_cf_mat[:,1] ] = 1

        # filter item with no relation interaction more than or equal 5 interaction
        item_mask = torch.zeros( ( self.n_items, ) )

        for r in relation_mask.keys():
            item_mask += ( relation_mask[ r ] >= 5 ).to( torch.int )

        item_mask = item_mask > 0

        adj_mat = adj_mat[ :, item_mask ]

        # filter user with less than 5 interaction
        user_mask = torch.sum( adj_mat, dim=-1 ) >= 10
        adj_mat = adj_mat[ user_mask, : ]

        # recompute num user based on filter adj mat
        self.n_users = adj_mat.shape[0]
        self.n_items = adj_mat.shape[1]

        # perform leave one out validation
        val_interact, test_interact = self.leave_one_out( adj_mat, torch.ones( ( self.n_users, self.n_items ) ) )

        # remove validation interaction and test interaction from train adj_mat
        train_mask = torch.ones( ( self.n_users, self.n_items ) )
        train_mask[ val_interact[:,0], val_interact[:,1] ] = 0
        train_mask[ test_interact[:,0], test_interact[:,1] ] = 0

        one_entity_relation = list()

        # recreate interaction mat
        for r in interact_mapper.keys():
            interact_mapper[r] = interact_mapper[r][ :, item_mask ]

            # check if any attribute don't have items
            num_items_attribute = torch.sum( interact_mapper[r], dim=-1 ) > 0

            interact_mapper[r] = interact_mapper[r][ num_items_attribute, : ]
            entity_mapper[r] = entity_mapper[r][ num_items_attribute ]

            if entity_mapper[r].shape[0] == 1:
                one_entity_relation.append( r )

        # remove relation with only 1 entity
        for r in one_entity_relation:
            interact_mapper.pop( r )
            entity_mapper.pop( r )

        self.interact_mapper = interact_mapper
        self.entity_mapper = entity_mapper

        return val_interact, test_interact, adj_mat, train_mask

    def train_test_val_split_metapath( self, adj_mat : torch.Tensor, train_mask : torch.Tensor, interact_mapper : defaultdict ):
        relation_mask = self.create_mask( interact_mapper )

        metapath_dataset = defaultdict( lambda : dict() )

        for r in interact_mapper.keys():
            metapath_train_mask = torch.clone( train_mask ) * ( relation_mask[r] > 0 ).reshape( 1, -1 )

            val_interact, test_interact = self.leave_one_out( adj_mat, metapath_train_mask )

            metapath_train_mask[ val_interact[:,0], val_interact[:,1] ] = 0
            metapath_train_mask[ test_interact[:,0], test_interact[:,1] ] = 0

            metapath_dataset[ r ][ 'val_interact' ] = val_interact
            metapath_dataset[ r ][ 'test_interact' ] = test_interact
            metapath_dataset[ r ][ 'train_mask' ] = metapath_train_mask

        return metapath_dataset

if __name__ == '__main__':
    yelp_path = '/Users/nuttupoomsaitoh/Desktop/class/seminar/PantipRec/datasets/yelp2018/'
    save_path = '/Users/nuttupoomsaitoh/Desktop/class/seminar/PantipRec/process_datasets/yelp2018/'
    dataset = YelpPreprocess( yelp_path, save_path )

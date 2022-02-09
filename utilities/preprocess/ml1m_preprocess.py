import sys
import os
import torch

class Ml1mPreprocess():
    def __init__( self, dataset_dir : str, save_dataset_dir : str ):
        self.n_users = 6040
        self.n_items = 3952
        self.user_file = os.path.join( dataset_dir, 'users.dat' )
        self.item_file = os.path.join( dataset_dir, 'movies.dat' )
        self.interaction_file = os.path.join( dataset_dir, 'ratings.dat' )

        self.adj_mat = self.load_cf()
        self.val_interact, self.test_interact, self.train_mask = self.train_test_val_split( self.adj_mat )
        self.item_genre, self.user_age, self.user_jobs = self.load_kg()
        self.adj_mat = ( self.adj_mat > 0 ).to( torch.int )

        self.save_process_data( save_dataset_dir )

    def save_process_data( self, process_dir ):
        if os.path.exists( process_dir ):
            print(f'please remove { process_dir }')
            sys.exit()

        os.mkdir( process_dir )
        os.chdir( process_dir )

        torch.save( self.val_interact, 'val_dataset.pt' )
        torch.save( self.test_interact, 'test_dataset.pt' )
        torch.save( self.adj_mat, 'train_adj_mat.pt' )
        torch.save( self.train_mask, 'train_mask.pt' )

        os.mkdir( os.path.join( process_dir, 'relation_mat' ) )
        os.chdir( os.path.join( process_dir, 'relation_mat' ) )

        for r in range( 3 ):
            os.mkdir( str( r ) )
            torch.save( self.item_genre, os.path.join( str(r), f'interact.pt' ) )

    def load_cf( self ):
        '''
        create interaction matrix where first column is user_id and second column is item_id
        '''
        adj_mat = torch.zeros( ( self.n_users, self.n_items ) )
        with open( self.interaction_file, 'r', encoding='iso-8859-1' ) as f:
            for line in f:
                user_id, item_id, _, timestamps = line.split('::')
                user_id, item_id, timestamps = int( user_id ) - 1, int( item_id ) - 1, int( timestamps )
                adj_mat[ user_id, item_id ] = timestamps

        return adj_mat

    def load_kg( self ):
        all_genre = [
            'Action',
            'Adventure',
            'Animation',
            "Children's",
            'Comedy',
            'Crime',
            'Documentary',
            'Drama',
            'Fantasy',
            'Film-Noir',
            'Horror',
            'Musical',
            'Mystery',
            'Romance',
            'Sci-Fi',
            'Thriller',
            'War',
            'Western'
        ]

        item_genre = torch.zeros( ( len( all_genre ), self.n_items ) )

        with open( self.item_file, encoding='iso-8859-1' ) as f:
            for line in f:
                movie_id, title, genre = line.strip().split('::')
                movie_id, genres = int( movie_id ) - 1, genre.split('|')
                for genre in genres:
                    item_genre[ all_genre.index( genre ), movie_id ] = 1

        age_mapper =[ 
             '1',   #Under 18
            '18',   #18-24
            '25',   #25-34
            '35',   #35-44
            '45',   #45-49
            '50',   #50-55
            '56',   #56+
        ]

        jobs_mapper = [
             '0',  #"other" or not specified
             '1',  #"academic/educator"
             '2',  #"artist"
             '3',  #"clerical/admin"
             '4',  #"college/grad student"
             '5',  #"customer service"
             '6',  #"doctor/health care"
             '7',  #"executive/managerial"
             '8',  #"farmer"
             '9',  #"homemaker"
            '10',  #"K-12 student"
            '11',  #"lawyer"
            '12',  #"programmer"
            '13',  #"retired"
            '14',  #"sales/marketing"
            '15',  #"scientist"
            '16',  #"self-employed"
            '17',  #"technician/engineer"
            '18',  #"tradesman/craftsman"
            '19',  #"unemployed"
            '20',  #"writer"

        ]
        user_age = torch.zeros( ( len( age_mapper ), self.n_users ) )
        user_jobs = torch.zeros( ( len( jobs_mapper ), self.n_users ) )

        with open( self.user_file, encoding='iso-8859-1' ) as f:
            for line in f:
                user_id, sex, age, job, zip_code = line.strip().split('::')
                user_id, age_id, job_id = int( user_id ) - 1, age_mapper.index( age ), jobs_mapper.index( job )

                user_age[ age_id, user_id ] = 1
                user_jobs[ job_id, user_id ] = 1

        return item_genre, user_age, user_jobs

    def leave_one_out( self, interaction_adj_mat : torch.Tensor, mask : torch.Tensor ):
        _, val_test_items = torch.topk( interaction_adj_mat * mask, 2 )
        neg_val_test_items = torch.multinomial( ( 1 - ( interaction_adj_mat > 0 ).to( torch.int ) ) * mask, num_samples=200 )

        pos_val_items, pos_test_items = torch.hsplit( val_test_items, sections=2 )
        neg_val_items, neg_test_items = torch.hsplit( neg_val_test_items, sections=2 )

        val_items = torch.hstack( ( pos_val_items, neg_val_items ) ).reshape( -1, 1 )
        test_items = torch.hstack( ( pos_test_items, neg_test_items ) ).reshape( -1, 1 )

        user_ids = torch.arange( self.n_users ).reshape( -1, 1 ).tile( 1, 101 ).reshape( -1, 1 )

        return torch.hstack( ( user_ids, val_items ) ), torch.hstack( ( user_ids, test_items ) )

    def train_test_val_split( self, adj_mat ):
        # perform leave one out validation
        val_interact, test_interact = self.leave_one_out( adj_mat, torch.ones( ( self.n_users, self.n_items ) ) )

        # remove validation interaction and test interaction from train adj_mat
        train_mask = torch.ones( ( self.n_users, self.n_items ) )
        train_mask[ val_interact[:,0], val_interact[:,1] ] = 0
        train_mask[ test_interact[:,0], test_interact[:,1] ] = 0

        return val_interact, test_interact, train_mask

if __name__ == '__main__':
    ml1m_path = '../../datasets/ml-1m/'
    ml1m_save_path = '../../process_datasets/ml-1m/'

    Ml1mPreprocess( ml1m_path, ml1m_save_path )

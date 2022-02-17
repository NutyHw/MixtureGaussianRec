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
        self.item_genre, self.user_age, self.user_jobs = self.load_kg()
        self.train_mask, self.val_mask, self.test_mask = self.train_test_val_split()

        self.save_process_data( save_dataset_dir )

    def save_process_data( self, process_dir ):
        if os.path.exists( process_dir ):
            print(f'please remove { process_dir }')
            sys.exit()

        os.mkdir( process_dir )
        os.chdir( process_dir )

        torch.save( self.adj_mat, 'adj_mat.pt' )
        torch.save( self.train_mask, 'train_mask.pt' )
        torch.save( self.val_mask, 'val_mask.pt' )
        torch.save( self.test_mask, 'test_mask.pt' )
        torch.save( { 
            'item_genre' : self.item_genre,
            'user_age' : self.user_age,
            'user_jobs' : self.user_jobs
        }, 'interact.pt' )

    def load_cf( self ):
        '''
        create interaction matrix where first column is user_id and second column is item_id
        '''
        adj_mat = torch.zeros( ( self.n_users, self.n_items ) )
        with open( self.interaction_file, 'r', encoding='iso-8859-1' ) as f:
            for line in f:
                user_id, item_id, _, _ = line.split('::')
                user_id, item_id = int( user_id ) - 1, int( item_id ) - 1
                adj_mat[ user_id, item_id ] = 1

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

    def train_test_val_split( self ):
        train_mask = torch.ones( ( self.n_users, self.n_items ) )
        val_mask = torch.zeros( ( self.n_users, self.n_items ) )
        test_mask = torch.zeros( ( self.n_users, self.n_items ) )

        val_test_samples = ( torch.sum( self.adj_mat, dim=-1 ) * 0.2 ).to( torch.int ).tolist()

        for user in range( self.n_users ):
            val_item_idx = torch.multinomial( self.adj_mat[ user ], num_samples=val_test_samples[ user ] )
            train_mask[ user, val_item_idx ] = 0
            val_mask[ user, val_item_idx ] = 1
            test_item_idx = torch.multinomial( self.adj_mat[ user ] * train_mask[ user ], num_samples=val_test_samples[ user ] )
            train_mask[ user, test_item_idx ] = 0
            test_mask[ user, test_item_idx ] = 1

        val_mask[ self.adj_mat == 0 ] = 1
        test_mask[ self.adj_mat == 0 ] = 1

        return train_mask, val_mask, test_mask

if __name__ == '__main__':
    ml1m_path = '../../datasets/ml-1m/'
    ml1m_save_path = '../../process_datasets/ml-1m/'

    Ml1mPreprocess( ml1m_path, ml1m_save_path )

import torch
from utilities.dataset.yelp_dataset import YelpDataset
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class Scheduler( pl.Callback ):
    def on_train_epoch_end( self, trainer, pl_module ):
        pl_module.dataset.samples()

class GeneralDataLoader( pl.LightningDataModule ):
    def __init__( self, r : int, batch_size : int ):
        self.batch_size = batch_size
        self.dataset = YelpDataset( r )

        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.n_categories = self.dataset.get_reg_mat().shape[0]

    def category_reg( self ):
        return self.dataset.get_reg_mat()

    def train_dataloader( self ):
        return DataLoader( 
            self.dataset,
            self.batch_size
        )

    def val_dataloader( self ):
        x = self.dataset.get_val()
        y = torch.zeros( ( x.shape[0] // 101, 101 ) )
        y[ :, 0 ] = 1
        y = y.reshape( -1, 1 )

        return DataLoader(
            TensorDataset( x[:101], y[:101] ),
            self.batch_size,
            shuffle=False
        )

    def test_dataloader( self ):
        x = self.dataset.get_test()
        y = torch.zeros( ( x.shape[0] // 101, 101 ) )
        y[ :, 0 ] = 1
        y = y.reshape( -1, 1 )

        return DataLoader(
            TensorDataset( x[:101], y[:101] ),
            self.batch_size,
            shuffle=False
        )

if __name__ == '__main__':
    dataloader = GeneralDataLoader( 0, 512 )
    train_dataloader = dataloader.val_dataloader()
    print( next( iter( train_dataloader ) ) )

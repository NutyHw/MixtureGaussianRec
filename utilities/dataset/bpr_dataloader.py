import torch
from utilities.dataset.yelp_dataset import YelpDataset
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class Scheduler( pl.Callback ):
    def on_train_epoch_end( self, trainer, pl_module ):
        pl_module.dataset.create_interact()

if __name__ == '__main__':
    dataloader = GeneralDataLoader( 0, 512 )
    train_dataloader = dataloader.val_dataloader()
    print( next( iter( train_dataloader ) ) )

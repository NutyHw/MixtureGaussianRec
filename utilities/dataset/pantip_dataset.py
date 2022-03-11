import  torch
from torch_cluster import  random_walk
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule

class PantipDataset( LightningDataModule ):
    def __init__( self, file, walk_length=7, context=2, p=1, q=2, batch_size=32 ):
        data = torch.load( file )
        print( data.keys() )
        self.train_adj_idx = data['train_g_index']
        self.val_adj_index = data['val_g_index']
        self.test_adj_index = data['test_g_index']
        self.num_user = len( data['unique_user_id'] )
        self.num_item = len( data['unique_item_id'] )
        self.num_nodes = len( data['unique_user_id'] ) + len( data['unique_item_id'] )
        self.batch_size = batch_size

        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.context = context

        self.val_score = torch.zeros( ( self.num_user, self.num_item ) )
        self.test_score = torch.zeros( ( self.num_user, self.num_item ) )
        self.val_mask = torch.ones( ( self.num_user, self.num_item ) )
        self.test_mask = torch.ones( ( self.num_user, self.num_item ) )

        self.val_score[ self.val_adj_index[0], self.val_adj_index[1] - self.num_user ] = 1
        self.test_score[ self.test_adj_index[0], self.test_adj_index[1] - self.num_user ] = 1

        self.val_mask[ data['train_user_item_index'][0], data['train_user_item_index'][1] ] = 0
        self.test_mask[ data['train_user_item_index'][0], data['train_user_item_index'][1] ] = 0

        self.reg_mat = dict()
        self.reg_mat['rooms'] = torch.zeros( ( self.num_item, len( data['all_rooms'] ) ) )
        self.reg_mat[ data[ 'item_room_adj_index' ][0], data['item_room_adj_index'][1] ] = 1

    def get_reg_mat( self, key ):
        return  self.reg_mat[ key ]

    def get_val( self ):
        return self.val_mask > 0, self.val_score

    def get_test( self ):
        return self.test_mask > 0, self.test_score

    def train_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.num_user ).reshape( 1, -1 ) ), batch_size=self.batch_size, collate_fn=self.seperate_relation )

    def val_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.num_user ).reshape( 1, -1 ) ), shuffle=False )

    def test_dataloader( self ):
        return DataLoader( TensorDataset( torch.arange( self.num_user ).reshape ( 1, -1 ) ), shuffle=False )

    def seperate_relation( self, batch):
        pos_sample, neg_sample = self.sample( batch )
        relation_type = pos_sample < self.num_user
        
        pos_user_item_relation_mask = torch.logical_xor( relation_type[:,0], relation_type[:,1] )

        relation_type = neg_sample < self.num_user
        neg_user_item_relation_mask = torch.logical_xor( relation_type[:,0], relation_type[:,1] )

        pos_sample, neg_sample = pos_sample[ pos_user_item_relation_mask ], neg_sample[ neg_user_item_relation_mask ]

        pos_is_user = pos_sample[:,0] < self.num_user
        neg_is_user = neg_sample[:,0] < self.num_user

        temp = pos_sample[~pos_is_user][:,0]
        pos_sample[~pos_is_user][:,0] = pos_sample[~pos_is_user][:,1]
        pos_sample[~pos_is_user][:,1] = temp

        temp = neg_sample[~neg_is_user][:,0]
        neg_sample[~neg_is_user][:,0] = neg_sample[~neg_is_user][:,1]
        neg_sample[~neg_is_user][:,1] = temp

        size = min( pos_sample.shape[0], neg_sample.shape[0] )

        return pos_sample[:size], neg_sample[:size]

    def sample( self, batch ):
        return self.pos_sample( batch[0][0] ), self.neg_sample( batch[0][0] )

    def pos_sample( self, batch ):
        rw = random_walk( self.train_adj_idx[0], self.train_adj_idx[1], batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch

        rw = torch.randint( self.num_nodes,
                           (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context])
        return torch.cat(walks, dim=0)

# if __name__ == '__main__':
    # val_mask, val_score = pantip.get_val()
    # num_interact = torch.sum( val_score , dim=-1 )
    # loader = pantip.train_dataloader()
    # print( ( torch.sum( pantip.get_reg_mat(), dim=1 ) > 0 ).all() )
    # # for i, interact in enumerate( loader ):
        # # print( interact[0].shape )


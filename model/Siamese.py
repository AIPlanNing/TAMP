import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.nn.modules.linear import Linear
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_sequence,
    pad_packed_sequence,
    pack_sequence
)
from .utils import get_mask_for_transformer


class ContrasitiveLoss(nn.Module):

    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.register_buffer('temperature', torch.tensor(temperature).to(device))
        self.register_buffer('negatives_mask', (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)))

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)

        return loss


class SiameseNet(nn.Module):

    def __init__(self, embd_size, gather_type='gru'):
        super().__init__()
        
        if gather_type == 'gru':
            self.gather = nn.GRU(
                embd_size,
                embd_size,
                batch_first=True,
                
            )

        elif gather_type == 'transformer':
            tsfm_enc_layer = nn.TransformerEncoderLayer(embd_size, 8)
            self.gather = nn.TransformerEncoder(tsfm_enc_layer, num_layers=2)

        else:
            raise NotImplementedError()



    def forward(self, x, y):
        if isinstance(self.gather, nn.GRU):
            _, hid_x = self.gather(x)
            _, hid_y = self.gather(y)
            hid_x = hid_x[0]
            hid_y = hid_y[0]
        elif isinstance(self.gather, nn.TransformerEncoder):
            pad_x, mask_x = get_mask_for_transformer(pack=x)
            pad_y, mask_y = get_mask_for_transformer(pack=y)
            mask_x = mask_x.to(pad_x.device)
            mask_y = mask_y.to(pad_y.device)
            pad_x, pad_y = pad_x.permute(1, 0, 2), pad_y.permute(1, 0, 2)
            hid_x, hid_y = self.gather(pad_x, src_key_padding_mask=mask_x).permute(1, 0, 2), self.gather(pad_y, src_key_padding_mask=mask_y).permute(1, 0, 2)
            hid_x, hid_y = hid_x.mean(dim=1), hid_y.mean(dim=1)
            
        return hid_x, hid_y

    @staticmethod
    def loss_siamese(x, y, label=None, margin=1.0):
        shuffle = torch.rand(1) > 0.5
        if label is not None:
            if shuffle:
                perm = torch.randperm(x.size()[0])
                x = x[perm]
                label_x = label[perm]
                label_y = label
                label = label_x == label_y
                label = label.to(torch.long)
            else:
                label = torch.ones_like(label)
        else:
            label = 0 if shuffle else 1
        ret = (1 - label) * torch.pow(F.pairwise_distance(x, y), 2) + \
        label * torch.pow(torch.clamp(margin - F.pairwise_distance(x,y), min=0.0), 2)
        return torch.mean(ret)
        

    
    def loss_infoNCE(self, x, y, label=None):
        pass



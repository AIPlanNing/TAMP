from struct import pack
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


from torch.nn.init import kaiming_uniform_
import math

class VAEBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _get_encoder(self, key=None):
        raise NotImplementedError(f'encoder should be rewrite.')

    def _get_decoder(self, key=None):
        raise NotImplementedError(f'decoder should be rewrite.')


    def encode(self, *args):
        raise NotImplementedError()

    def decode(self, *args):
        raise NotImplementedError()

    def loss_reconstruct(self, x, x_recon):
        loss =  -1.0 * torch.sum(x*torch.log_softmax(x_recon, dim=1))
        #return loss
        return loss / len(x)

    def loss_KLD(self, mu, log_sigma):
        kld =  -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        #return kld
        return kld / len(mu)


# VAE model
class VAE(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout=0.0, nonlin='relu', hid_size=64, **kwargs):


        super(VAE, self).__init__()
        self.hid_size = hid_size
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.nonlin = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid}[nonlin]

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hid_size*2),
            self.nonlin(),
            nn.Linear(hid_size*2, hid_size),
            self.nonlin(),
            
        )


        self.fc_mu = nn.Linear(hid_size, num_topics)
        self.fc_logvar = nn.Linear(hid_size, num_topics)

        self.decoder = nn.Sequential(
            nn.Linear(num_topics, hid_size),
            self.nonlin(),
            nn.Linear(hid_size, vocab_size)
        )

        self.latent_dim = num_topics
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(num_topics, num_topics)
        
        
    def encode(self, x):
        hid = self.encoder(x)
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, log_var

    def inference(self,x):
        mu, log_var = self.encode(x)
        theta = torch.softmax(x,dim=1)
        return theta
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        hid = self.decoder(z)
        return hid
    
    def forward(self, x, collate_fn=None):
        mu, log_var = self.encode(x)
        _theta = self.reparameterize(mu, log_var)
        _theta = self.fc1(_theta) 
        '''if collate_fn!=None:
            theta = collate_fn(_theta)
        else:
            theta = _theta'''
        theta = F.softmax(_theta, dim=-1)
        x_reconst = self.decode(theta)
        return mu, log_var, x_reconst, theta

    def loss_reconstruct(self, x, x_recon):
        '''
        重构损失，x和x_recon的交叉熵损失
        :param x(torch.Tensor): 输入句子的词袋表示
        :param x_recon(torch.Tensor): 重构词袋
        '''
        return -1.0 * torch.sum(x*torch.log_softmax(x_recon, dim=1)) / len(x)

    def loss_KLD(self, mu, log_sigma):
        '''
        KLD loss KLD(N(mu, sigma) || N(0, 1))

        '''
        return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp()) / len(mu)

class EVAE(VAE):
    def __init__(self, vocab_size, num_topics, embd_size, dropout=0.0, nonlin='relu', hid_size=64,):
        super(EVAE,self).__init__(vocab_size, num_topics, dropout, nonlin, hid_size)
        self.emb_size = embd_size
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        #self.word_embd = nn.Linear(embd_size, self.vocab_size)
        self.word_embd = nn.Embedding(vocab_size, embd_size)
        kaiming_uniform_(self.word_embd.weight, a=math.sqrt(5))
        #self.topic_embd = nn.Linear(embd_size, self.num_topics)
        self.topic_embd = nn.Embedding(num_topics, embd_size)
        kaiming_uniform_(self.topic_embd.weight, a=math.sqrt(5))
        self.decoder = None

    def decode(self,z):
        #wght_dec = self.topic_embd(self.word_embd.weight) #[K,V]
        #beta = F.softmax(wght_dec,dim=0).transpose(1,0)
        wght_dec = torch.mm(self.word_embd.weight, self.topic_embd.weight.transpose(1, 0))
        beta = F.softmax(wght_dec, dim=0).transpose(1, 0)
        res = torch.mm(z,beta)
        # logits = res
        logits = torch.log(res+1e-6)
        return logits



class EEVAE(EVAE):
    def __init__(self, vocab_size, num_topics, embd_size, dropout=0, nonlin='relu', hid_size=128):
        super().__init__(vocab_size, num_topics, embd_size, dropout=dropout, nonlin=nonlin, hid_size=hid_size)
        self.rnn = nn.GRU(embd_size, embd_size)
        self.fc_rnn = nn.Linear(embd_size, hid_size)
    def encode(self, x):
        hid = self.get_hid(x)
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        
        return mu, log_var

    def get_hid(self, x):
        if isinstance(x, PackedSequence):
            x, x_len = pad_packed_sequence(x, batch_first=True)
        else:
            x_len = [len(i) for i in x]
        embd = self.word_embd(x)
        embd_pack = pack_padded_sequence(embd, x_len, batch_first=True, enforce_sorted=False)
        _, hid = self.rnn(embd_pack)
        hid = hid.squeeze()
        hid = self.fc_rnn(hid)
        return hid

    

    

class StateNet(VAEBase):

    def __init__(self, num_action, vocab_size, embd_size=256, word_embd=None, enc_type='transformer') -> None:
        super().__init__()
        self.num_action = num_action
        self.vocab_size = vocab_size
        self.embd_size = embd_size

        self.cls = nn.Embedding(1, embd_size)
        self.prestate_embd = nn.Linear(embd_size, num_action)
        self.word_embd = nn.Linear(embd_size, vocab_size) if word_embd is None else word_embd


        self.encoder = self._get_encoder(key=enc_type)
        self.mu = nn.Linear(embd_size, num_action)
        self.log_sigma = nn.Linear(embd_size, num_action)
        self.fc = nn.Linear(num_action, num_action)

        self.action_predictor = nn.Sequential(
            nn.Linear(self.embd_size, num_action),
            nn.Softmax(dim=1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.ce = nn.CrossEntropyLoss()
        

    def _get_encoder(self, **kwargs):
        encoder_type = kwargs.get('key', 'transformer')
        if encoder_type == 'gru':
            enc = nn.GRU(
                self.embd_size,
                self.embd_size,
                batch_first=True
            )
            
        elif encoder_type == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(d_model=self.embd_size, nhead=8, dim_feedforward=256)
            enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        else:
            raise NotImplementedError(f'encoder type "{encoder_type}" is not supported yet.')
        return enc

    def _get_decoder(self):
        pass

    def encode_rnn(self, pack):
        assert isinstance(pack, PackedSequence)
        ret, hid = self.encoder(pack)
        z = hid[0]
        mu = self.mu(z)
        log_sigma = self.log_sigma(z)
        return mu, log_sigma, z

    def encode_transformer(self, pack):
        pad, lengths = pad_packed_sequence(pack, batch_first=True) # padding_value
        mask = torch.ones((pad.shape[0], pad.shape[1])).to(torch.bool)
        for row_idx, length in enumerate(lengths):
            for column_idx in range(length):
                mask[row_idx][column_idx] = False
        batch = pad
        batch_size, seq_size, embd_size = batch.shape[0], batch.shape[1], batch.shape[2]
        cls = self.__dict__.get('cls', torch.randn_like(batch[0][0]))
        # cls = self.cls(torch.zeros_like((batch_size, 1), device=batch.device))
        cls = torch.stack([cls] * batch_size).unsqueeze(dim=1)
        cls = cls.to(batch.device)

        cls_mask = torch.zeros((batch_size, 1)).bool()
        mask = torch.cat((cls_mask, mask), dim=1)
        mask = mask.to(batch.device)

        batch = torch.cat((cls, batch), dim=1).permute(1, 0, 2)

        z = self.encoder(batch, src_key_padding_mask=mask) 
        z = z.permute(1, 0, 2) #(N, L+1, E)
        z = z[:, 0] #(N, E)
        mu = self.mu(z)
        log_sigma = self.log_sigma(z)
        return mu, log_sigma, z


    def encode(self, batch, mask=None):
        
        if isinstance(self.encoder, nn.GRU):
            return self.encode_rnn(batch)
        elif isinstance(self.encoder, nn.TransformerEncoder):
            return self.encode_transformer(batch)
        

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        '''logits = torch.mm(self.prestate_embd.weight, self.word_embd.weight.transpose(1, 0)).softmax(dim=0)
        logits = torch.mm(z, logits)
        return logits'''
        wght_dec = self.prestate_embd(self.word_embd.weight) #[K,V]
        beta = F.softmax(wght_dec,dim=0).transpose(1,0)
        res = torch.mm(z,beta)
        # logits = res
        logits = torch.log(res+1e-6)
        return logits




    def forward(self, batch, mask=None):
        '''
        1. CE of prestate_dist and prestate label + reconstruction from prestate_state to states
        2. logits of prestate and word, reconstruct to states labels
        :param batch(torch.tensor): (N * S * E) N for batch size, S for number of sentences, 
        E for embedding size of topic
        '''
        mu, log_sigma, state_hid = self.encode(batch, mask)
        prediction = self.action_predictor(state_hid)
        z = self.reparameterize(mu, log_sigma)
        z = self.fc(z)
        prestate_dist = self.softmax(z)
        recon = self.decode(prestate_dist)
        
        # return mu, log_sigma, recon, prestate_dist
        return {
            'mu': mu,
            'log_sigma': log_sigma,
            'recon': recon,
            'prestate_dist': prestate_dist,
            'action_prediction': prediction

        }

    def loss_CE(self, x, y):
        loss = self.ce(x, y)
        return loss




if __name__ == '__main__':
    pass

        

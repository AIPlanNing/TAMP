from unicodedata import bidirectional
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from functools import reduce, partial
from torch.nn.modules import dropout
from torch.nn.modules.linear import Linear
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, \
    pack_padded_sequence, pack_sequence, PackedSequence
import torch.nn.functional as F
import itertools
from gensim.models import word2vec
import time
import random




class ModelOutputForPlan():
    '''
    生成solution文件所需要的输入
    '''
    def __init__(self, data: dict):
        '''self.data = data
        self.id = self.data['id']
        self.predicate = self.data['predicate']
        self.extract_params = self.data['extract_params']
        self.gold_params = self.data['gold_params']
        self.label = self.data['label']
        self.sentence = self.data['sentence']
        self.is_goal = data['is_goal']'''

        self.data = data
        self.id = self.data.get('id')
        self.predicate = self.data.get('predicate')
        self.extract_params = self.data.get('extract_params')[:]
        self.gold_params = self.data.get('gold_params')
        self.label = self.data.get('label')
        self.sentence = self.data.get('sentence')
        self.is_goal = data.get('is_goal')
        self.args_type = data.get('args_type')
        if not self.args_type:
            Warning('no args_type.')

    def __le__(self, y):
        return self.id <= y.id

    def __repr__(self) -> str:
        return f'{self.id} {self.predicate} {self.gold_params}'




class MappingModel(torch.nn.Module):
    '''
    文本映射到词表.
    
    '''

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        if args.embedding_from_wv:
            self.embedding = self.get_embedding_layer_from_w2v(args.wv_model_file, args.dataset.word2id, freeze=args.embedding_freeze)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embd_hid)
        # self.param_embdding = nn.Embedding(args.params_num, args.embd_hid)
        self.encoder = self.get_encoder('gru')
        self.enc_act = nn.ReLU()
        self.pred_net = self.get_pred_net()
        self.sort_net = self.get_sort_net('o')
        self.sort_net_act = nn.ReLU()

    def get_embedding_layer_from_w2v(self, w2v_model_file, dataset_word2id, freeze):
        model = word2vec.Word2Vec.load(w2v_model_file)
        wv = model.wv
        vocab_size = len(dataset_word2id.keys())
        weight = torch.randn((vocab_size, wv.vector_size))
        for word, idx in wv.key_to_index.items():
            weight[dataset_word2id[word]] = torch.tensor(wv.vectors[idx])
        
        return nn.Embedding.from_pretrained(weight, padding_idx=0, freeze=freeze)

    def params_forward(self, params, hid):
        '''
        :param params(tuple): (flat_params, params_lens)
        :param hid: (batch, hid_size)
        
        '''

        ret = {
            'rep': [],
            'idx': []
        }
        # pack twice
        if isinstance(params, tuple):
            params, params_lens = params
        if isinstance(params, torch.nn.utils.rnn.PackedSequence):
            params, pad_params_len = pad_packed_sequence(params, batch_first=True)
        embd = self.embedding(params)
        packed_embd = pack_padded_sequence(embd, pad_params_len, batch_first=True, enforce_sorted=False)

        out, last_hid = self.sort_net(packed_embd)
        last_hid = last_hid[-1]
        _, sorted_indices = torch.sort(torch.tensor(params_lens), descending=True)
        sorted_indices = sorted_indices.to(last_hid.device)
        batch_sizes = torch.tensor((hid.size()[0], last_hid.size()[0]-hid.size()[0]), dtype=torch.int64)
        packed_last_hid = PackedSequence(last_hid, batch_sizes, sorted_indices)
        pad_last_hid, _ = pad_packed_sequence(packed_last_hid, batch_first=True)
        pad_last_hid = self.sort_net_act(pad_last_hid)
        hid = hid.unsqueeze(2)
        logits = torch.bmm(pad_last_hid, hid)
        logits = F.softmax(logits, dim=1)
        topk_val, topk_idx = logits.topk(1, dim=1)
        idx = torch.cat((topk_idx,)*pad_last_hid.size()[-1], 2)
        rep = pad_last_hid.gather(1, idx).squeeze()


        ret['rep'] = rep
        ret['idx'] = topk_idx.squeeze()

        return ret
            
    

    def forward(self, batch, params=None, return_dict=True):
        # TODO pack
        batch_size = batch.size()[0]
        embd = self.embedding(batch)
        _, hid = self.encoder(embd)
        hid = torch.cat((hid[-1], hid[-2]), 1)
        hid = self.enc_act(hid)
        


        ret = self.params_forward(params, hid)
        param_pred = ret['idx']
        param_hid = ret['rep']

        out = self.pred_net(hid)
        
        out = F.softmax(out/self.args.temperature, dim=1)
        # out = F.gumbel_softmax(out/self.args.temperature, dim=1)
        pred_hid = torch.mm(out, self.embedding.weight)

        if return_dict:
            return {
                'pred_out': out,
                'pred_hid': pred_hid,
                'params_hid': param_hid,
                'params_out': param_pred
            }
        return out, hid
    
    def get_params_candidate(self, params):
        ret = itertools.permutations(params, len(params))
        return list(ret)

    def get_encoder(self, enc_type='gru'):
        enc = None
        if enc_type == 'gru':
            enc = nn.GRU(
                self.args.embd_hid, 
                self.args.hidden, 
                num_layers=self.args.encoder_num,
                batch_first=True, 
                dropout=self.args.dropout,
                bidirectional=True
            )
        return enc
    
    def get_pred_net(self):
        pred_net = nn.Sequential(
            nn.Linear(self.args.hidden*2, self.args.vocab_size), 
            nn.Softmax(dim=1)
        )

        return pred_net

    def get_sort_net(self, key):
        if key == 'pn':
            return EncDec(self.args)
        rnn = nn.GRU(
            self.args.embd_hid,
            self.args.hidden * 2,
            # num_layers=self.args.encoder_num,
            batch_first=True,
            dropout=self.args.dropout,
            # bidirectional=True
        )
        
        return rnn

class StateModel(nn.Module):
    '''
    sentence to triplet
    multi triplet to state
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mm = MappingModel(self.args)
        self.state_encoder = self.get_state_encoder(key=args.state_encoder_type)
        
        self.pred_param_encoder = self.get_pred_param_encoder()

        self.decoder = nn.Linear(args.hidden, args.vocab_size)

    def get_pred_param_encoder(self):
        return nn.Sequential(
            nn.Linear(self.args.hidden*3, self.args.hidden*2),
            nn.Linear(self.args.hidden*2, self.args.hidden)
        )
        

    def get_state_encoder(self, key='avg'):
        if key == 'avg':
            ret = F.adaptive_avg_pool2d
        elif key == 'rnn':
            ret = nn.GRU(self.args.hidden, self.args.hidden)
        return ret

    def state_forward(self, hid, lengths):
        '''
        :param hid: tensor, pred_len * hid_size
        :param lengths: length of each state
        '''
        
        
        if self.args.state_encoder_type == 'rnn':
            _, sorted_indices = torch.sort(torch.tensor(lengths), descending=True)

            batch_sizes = []
            bs = sum([i > 0 for i in lengths])
            while bs:
                batch_sizes.append(bs)
                lengths = [i-1 for i in lengths]
                bs = sum([i > 0 for i in lengths])

            
            packed_hid = PackedSequence(hid, torch.tensor(batch_sizes), sorted_indices.to(hid.device))
            out, _ = self.state_encoder(packed_hid)
            ret = _[-1]
        elif self.args.state_encoder_type == 'avg':
            count = 0
            hid_size = hid.size()[1]
            ret = []

            for l in lengths:
                ret.append(self.state_encoder(hid[count: count+l].unsqueeze(0), (1, hid_size)).squeeze())
                count += l
            ret = torch.stack(ret)
        return ret

    def forward_once(self, batch, params, batch_type='sentence', **kwargs):
        '''
        :param batch (torch.tensor): sentences in size (batch, sent_len)
        :param params (torch.tensor | list): tensor in size (batch, max_param_len)
        if batch_type == 'predicate', list of params if batch_type == 'sentence'
        :param batch_type (str): indicate the type of input 
        '''
        ret = {}
        state_lens = kwargs.get('lengths')
        params_lens = kwargs.get('params_len')
        # params = reduce(lambda x, y: x + y, params)# MARK
        # state_lens = [len(i) for i in batch]
        # batch_sent = reduce(lambda x, y: x + y, batch)
        
        if batch_type == 'sentence':
            
            # pad_batch_sent = pad_sequence(batch, batch_first=True)
            sent_enc = self.mm(batch, params=params, return_dict=True)
            pred_hid, params_hid, pred_out, params_out = sent_enc['pred_hid'], sent_enc['params_hid'], sent_enc['pred_out'], sent_enc['params_out']
            ret['pred_out'] = pred_out
            ret['params_out'] = params_out
        elif batch_type == 'predicate':
            pred_hid = self.mm.embedding(batch)
            
            # pad_params = pad_sequence(params, batch_first=True)
            pad_params_embd = self.mm.embedding(params)
            pack_padded_params = pack_padded_sequence(pad_params_embd, lengths=[len(i) for i in params], batch_first=True, enforce_sorted=False)
            packed_params_hid, _ = self.mm.sort_net(pack_padded_params)
            pad_packed_params_hid = pad_packed_sequence(packed_params_hid, batch_first=True)
            params_hid, idx = pad_packed_params_hid
            # MARK
            params_hid = torch.stack([params_hid[i, j, :] for i, j in zip(range(len(params_hid)), idx-1)])
        else:
            raise NotImplementedError()
            
        
        pred_params_cat = torch.cat((pred_hid, params_hid), dim=1)
        pred_params_hid = self.pred_param_encoder(pred_params_cat)
        state_hid = self.state_forward(pred_params_hid, state_lens)
        ret['state_hid'] = state_hid
        return ret

    def forward(self, batch_x, batch_y=None, x_params=None, y_params=None, **kwargs):
        '''
        :params batch_x(torch.tensor): (sentences_num, sentence_len) sentences_num为一个
        batch中所有state所包含的句子总数，需要提供x_len参数指出每个state包含多少个句子
        :params batch_y(torch.tensor): (predicate_num, 1)predicate_num为一个batch中所有
        谓词的总数，需要提供y_len指出每个state中包含多少个谓词，注意：batch_y和batch_x可能
        不同shape
        :params x_params(tuple): x_params[0](torch.tensor)为参数的全排列，x_params[1]指出
        每个参数被扩展为多少个candidate
        
        '''
        x_len = kwargs.get('x_len')
        y_len = kwargs.get('y_len')

        x_state_ = self.forward_once(batch_x, x_params, batch_type='sentence', lengths=x_len)
        x_state, pred_out, params_out = x_state_['state_hid'], x_state_['pred_out'], x_state_['params_out']
        y_state = self.forward_once(batch_y, y_params, batch_type='predicate', lengths=y_len)['state_hid']
        return x_state, y_state, pred_out, params_out


    def predict(self, inp, params, param_candidates, topk=1):
        '''
        给定输入，给出模型预测的谓词以及参数顺序
        :params inp(torch.tensor): 输入的句子
        :params params(torch.tensor): 输入的参数
        '''
        ret = self.mm(inp, params, return_dict=True)
        predicate_logits = ret['pred_out']
        params = ret['params_out'].tolist()
        params = [candd[idx] for (candd, idx) in zip(param_candidates, params)]
        predicate = predicate_logits.topk(topk)[1]

        return predicate, params
    def predict_sentence(self, sent, word2id):
        if isinstance(sent, str):
            tokens = sent.split(' ')
        else:
            tokens = sent
        tokens = [word2id[i] for i in tokens]
        tokens = torch.tensor(tokens)
        with torch.no_grad():
            embd = self.mm.embedding(tokens)
            enc, _ = self.mm.encoder(embd)
            enc = self.mm.enc_act(enc)

    def info_nce_loss(self, x, y):

        pass
    
    def reconstruct_contra_loss(self, state_hid_x, state_hid_y, y, y_len, ratio=0.5):
        con_loss = self.contrasitive_loss(state_hid_x, state_hid_y)
        recon_loss = self.reconstruct_loss(state_hid_x, y, y_len)
        return ratio * con_loss + (1 - ratio) * recon_loss, con_loss, recon_loss

    def calculate_loss(self, state_hid_x, state_hid_y, sentences_logit, sentences_label, ratio=0.5, **kwargs):
        if not (isinstance(sentences_logit, tuple), isinstance(sentences_label, tuple)):
            raise ValueError('input of bow loss should be tuple of (input, input_state_len)')
        loss_contra = self.contrasitive_loss(state_hid_x, state_hid_y)
        x, x_len = sentences_logit
        y, y_len = sentences_label
        
        if ratio > 1 or ratio < 0:
            raise ValueError(f'ratio should between 0 and 1, but got {ratio}.')
        loss_bow = self._bow_loss(x, x_len, y, y_len, **kwargs)
        loss = ratio * loss_contra + loss_bow
        return loss, ratio * loss_contra, loss_bow
    
    

    def reconstruct_loss(self, x_state, y, y_len):
        pred = self.decoder(x_state)
        label_prob = torch.zeros_like(pred)
        x_idx = 0
        y_idx = 0
        for _ylen in y_len:
            temp_tensor = torch.zeros_like(pred[0])
            for item in y[y_idx: y_idx+_ylen]:
                temp_tensor[item.item()] += 1
            y_idx += _ylen
            label_prob[x_idx] = temp_tensor.softmax(dim=-1)
            x_idx += 1
        return F.kl_div(pred.softmax(dim=1).log(), label_prob, reduction='batchmean')
        
    def kb_loss(self, x, x_len, y, y_len, **kwargs):
        x_params = kwargs.get('x_params')
        y_params = kwargs.get('y_params')
        target = torch.zeros(x.size()[0], dtype=torch.long)
        g_y_idx, g_x_idx = 0, 0
        for _xlen, _ylen in zip(x_len, y_len):
            for _ in range(_xlen):
                temp = []
                
                for y_idx in range(g_y_idx, g_y_idx + _ylen):
                    if set(x_params[g_x_idx]) == set(y_params[y_idx]):
                        temp.append(y[y_idx].item())

                target[g_x_idx] = torch.tensor(random.choice(temp), dtype=torch.long)
                g_x_idx += 1
                
            g_y_idx += _ylen
        target = target.to(x.device)

        return F.cross_entropy(x, target)

    def _bow_loss(self, x, x_len, y, y_len, **kwargs):
        '''
        根据参数进行过滤
        '''
        x_params = kwargs.get('x_params')
        y_params = kwargs.get('y_params')
        label_prob = torch.empty_like(x)
        g_y_idx, g_x_idx = 0, 0
        for _xlen, _ylen in zip(x_len, y_len):
            for _ in range(_xlen):
                temp_tensor = torch.zeros_like(x[0])
                for y_idx in range(g_y_idx, g_y_idx + _ylen):
                    if set(x_params[g_x_idx]) == set(y_params[y_idx]):
                        temp_tensor[y[y_idx].item()] = 1
                label_prob[g_x_idx] = temp_tensor
                g_x_idx += 1
                
            g_y_idx += _ylen

        return F.kl_div(x.log(), F.softmax(label_prob, dim=1), reduction='batchmean')

    def bow_loss(self, x, x_len, y, y_len):
        label_prob = torch.empty_like(x)
        g_y_idx = 0
        g_x_idx = 0
        for _xlen, _ylen in zip(x_len, y_len):
            temp_tensor = torch.zeros_like(x[0])
            for item in y[g_y_idx: g_y_idx + _ylen]:
                temp_tensor[item.item()] = 1
            g_y_idx += _ylen

            label_prob[g_x_idx: g_x_idx + _xlen] = temp_tensor
            g_x_idx += _xlen
        
        return F.kl_div(x.log(), F.softmax(label_prob, dim=1), reduction='batchmean')

    def contrasitive_loss(self, x, y, margin=2.0):
        shuffle = torch.rand(1) > 0.5
        label = 0
        if shuffle:
            perm = torch.randperm(x.size()[0])
            x = x[perm]
            label = 1
        loss = self._contrasitive_loss(x, y, label=label, margin=margin)
        return loss

    def _contrasitive_loss(self, x, y, label=0, margin=2.0):
        ret = (1 - label) * torch.pow(F.pairwise_distance(x, y), 2) + \
        label * torch.pow(torch.clamp(margin - F.pairwise_distance(x,y), min=0.0), 2)
        return torch.mean(ret)

class SimpleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embd_hid)
        self.ln_embd = nn.LayerNorm(self.args.embd_hid)
        self.encoder = nn.GRU(
            args.embd_hid, 
            args.hidden, 
            num_layers=args.encoder_num, 
            batch_first=True, 
            dropout=args.dropout, 
            bidirectional=True
        )
        self.ln_encoder = nn.LayerNorm(self.args.hidden*2)
        self.mlp = nn.Linear(
            args.hidden*2,
            args.n_catagory
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        embd = self.embedding(batch)
        embd = self.ln_embd(embd)
        hid, _ = self.encoder(embd)
        hid = hid[:, -1, :]
        hid = self.ln_encoder(hid)
        hid = self.mlp(hid)
        out = self.softmax(hid)
        return out



class EncDec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc = self.get_encoder()
        self.dec = self.get_decoder()


    def forward(self, batch):
        pack_enc_out, enc_hid = self.enc(batch)
        dec_hid, _ = self.dec(batch)
        out = F.linear(dec_hid)
        out = F.softmax(out)
        return out

    def get_encoder(self):
        enc = nn.GRU(
                self.args.embd_hid, 
                self.args.hidden, 
                # num_layers=self.args.encoder_num,
                batch_first=True, 
                dropout=self.args.dropout,
                bidirectional=False
            )
        return enc

    def get_decoder(self):
        dec = nn.GRU(
                self.args.embd_hid, 
                self.args.hidden, 
                # num_layers=self.args.encoder_num,
                batch_first=True, 
                dropout=self.args.dropout,
                bidirectional=False
            )

        dec = nn.GRUCell(
            self.args.embd_hid,
            self.args.hidden
        )
        
        return dec

    


class VAE(nn.Module):

    def __init__(self, args, topic_num=20):
        super().__init__()
        self.topic_num = topic_num
        self.args = args
        self.encoder = self.get_encoder()
        self.mu = nn.Linear(args.hidden, topic_num)
        self.log_sigma = nn.Linear(args.hidden, topic_num)
        self.decoder = self.get_decoder()
        self.mlp = nn.Linear(topic_num, topic_num)

    def get_encoder(self):
        enc = nn.Sequential(
            nn.Linear(self.args.vocab_size, self.args.hidden*2),
            nn.ReLU(),
            nn.Linear(self.args.hidden*2, self.args.hidden),
            nn.ReLU(),
            nn.Linear(self.args.hidden, self.args.hidden),
            nn.ReLU()
        )
        return enc

    def get_decoder(self, topic_num=20):
        dec = nn.Sequential(
            nn.Linear(topic_num, self.args.hidden),
            nn.ReLU(),
            nn.Linear(self.args.hidden, self.args.hidden*2),
            nn.ReLU(),
            nn.Linear(self.args.hidden*2, self.args.vocab_size),
            
            
        )
        return dec

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma / 2)
        eps = torch.rand_like(std)
        return mu + eps * std
        

    def encode(self, x):
        hid = self.encoder(x)
        mu, log_sigma = self.mu(hid), self.log_sigma(hid)
        return mu, log_sigma

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon
        

    def forward(self, batch):
        # print(batch.size(), self.args.dataset.vocab_size)
        mu, log_sigma = self.encode(batch)
        z = self.reparameterize(mu, log_sigma)
        z = self.mlp(z)
        z = F.softmax(z, dim=1)
        x_recon = self.decode(z)
        return mu, log_sigma, x_recon, z
    
    def reconstruct_loss(self, x, x_recon):
        # loss = F.pairwise_distance(x, x_recon)
        logsoftmax = torch.log_softmax(x_recon,dim=1)
        loss = -1.0 * torch.sum(x*logsoftmax)
        return loss

    def kl_loss(self, mu, log_sigma):
        loss = -0.5 * torch.sum(1+log_sigma-mu.pow(2)-log_sigma.exp())
        return loss

    def loss(self, x, x_recon, mu, log_sigma, beta=1.0):
        loss = self.reconstruct_loss(x, x_recon) + beta * self.kl_loss(mu, log_sigma)
        return loss

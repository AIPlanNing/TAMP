import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_sequence,
    pad_packed_sequence,
    pack_sequence
)

from model.ParamSorter import ParamSorter
from .vae import VAE, VAEBase, StateNet, EVAE, EEVAE
from .wae import WAE
import logging
logger = logging.getLogger(__name__)

import tqdm
from functools import reduce
import numpy as np



class NTM(nn.Module):

    def __init__(self, vocab_size, num_topics, num_action, embd_size=256, task='nb', model_type='evae', enc_type='transformer', hid_size=64, **kwargs) -> None:

        '''
        :param enc_type: state net encoder type. (gru | transformer)
        '''
        super().__init__()
        self.num_param_types = kwargs.get('num_param_types')
        self.task = task
        self.model_type = model_type
        prop_net_map = {
            'vae': VAE,
            'evae': EVAE,
            'wae': WAE,
            'eevae': EEVAE
        }

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_actions = num_action
        self.embd_size = embd_size

        self.prop_net: EEVAE = prop_net_map[model_type](vocab_size=vocab_size, num_topics=num_topics, embd_size=embd_size, hid_size=hid_size)
        feature_in = self.prop_net.fc_rnn.weight.shape[0]
        self.predict_head = PredictHead(feature_in, self.vocab_size)
        self.param_sorter = ParamSorter(self.prop_net.hid_size)
    
    def param_sort_forward(self, param_inp):
        param_inp = param_inp.to(self.prop_net.word_embd.weight.device)
        hid = self.prop_net.get_hid(param_inp)
        param_embd = self.prop_net.word_embd(param_inp)
        param_out = self.param_sorter(hid, param_embd)
        return param_out

    def param_sort_predict(self, param_inp):
        #param_inp (N, 2)
        param_inp = param_inp.to(self.prop_net.word_embd.weight.device)
        prediction = self.param_sort_forward(param_inp) #(N, 2, 2)
        topk = prediction.topk(1)[1].squeeze() #(N, 2)
        topk[:, 1] = (~(topk[:, 0].bool())).long()
        return param_inp.gather(1, topk)



    def predict_head_forward(self, batch):
        hid = self.prop_net.get_hid(batch)
        out = self.predict_head(hid)
        return out


    def forward(self, batch, **kwargs):
        '''
        :param batch(PackedSequence):
        :return: 返回的顺序according to batch.data的顺序
        '''

        out = self.prop_net(batch)
        

        if self.model_type in ['vae', 'evae', 'eevae']:
            prop_mu, prop_log_sigma, prop_recon, prop_topic_dist = out
        elif self.model_type in ['wae']:
            prop_recon, prop_topic_dist = out


        
        return {
            'prop_recon': prop_recon,
            'prop_dist': prop_topic_dist,
            'others': [prop_mu, prop_log_sigma]
        }

    def label_predict_loss(self, x, label, weight=None):
        return self.predict_head.loss_ce(x, label, weight)

    def topic2word(self, topic_id, topK=10, stop_word_ids: list=None):
        topic2wordidx = []
        topic_wordidx = self.show_topic_word_ids(topK)
        assert len(topic_wordidx) == self.num_topics
        for topic_word_list in topic_wordidx:
            for word_idx in topic_word_list:
                if not word_idx in stop_word_ids:
                    stop_word_ids.append(word_idx)
                    topic2wordidx.append(word_idx)
                    break
        return topic2wordidx if topic_id is None else topic2wordidx[topic_id]


    def predict_head_predict(self, *args, **kwargs):
        inp = args[0]
        inp = inp.to(self.prop_net.topic_embd.weight.device)
        out = self.predict_head_forward(inp)
        return out


    def predict(self, *args, **kwargs):
        stop_words = kwargs.get('stop_words', [])
        assert isinstance(stop_words, list)

        
        inp = args[0]
        inp = inp.to(self.prop_net.topic_embd.weight.device)
        
        
        self.eval()
        out = self(inp)
        topic_dist = out['prop_dist']
        topic_id = topic_dist.topk(1, dim=1)[1].squeeze().tolist()

        #pred = [self.topic2word(id, 20, stop_words[:]) for id in topic_id] # idx of predicates
        #ret[0] = pred
        
        return topic_id

    def show_topic_word_ids(self, topK=5):
        idxes = torch.eye(self.num_topics).to(self.prop_net.topic_embd.weight.device)
        word_dist = self.prop_net.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        vals,indices = torch.topk(word_dist,topK,dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        return indices

    def inference_by_bow(self,doc_bow):
        # doc_bow: torch.tensor [vocab_size]; optional: np.array [vocab_size]
        if isinstance(doc_bow,np.ndarray):
            doc_bow = torch.from_numpy(doc_bow)
        doc_bow = doc_bow.reshape(-1,self.bow_dim).to(self.device)
        with torch.no_grad():
            mu,log_var = self.vae.encode(doc_bow)
            mu = self.vae.fc1(mu) 
            theta = F.softmax(mu,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()


    def inference(self, doc_tokenized, dictionary,normalize=True):
        doc_bow = torch.zeros(1,self.bow_dim)
        for token in doc_tokenized:
            try:
                idx = dictionary.token2id[token]
                doc_bow[0][idx] += 1.0
            except:
                print(f'{token} not in the vocabulary.')
        doc_bow = doc_bow.to(self.device)
        with torch.no_grad():
            mu,log_var = self.vae.encode(doc_bow)
            mu = self.vae.fc1(mu)
            if normalize:
                theta = F.softmax(mu,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()

    def get_embed(self,train_data, num=1000):
        self.eval()
        data_loader = DataLoader(train_data, batch_size=512,shuffle=False, num_workers=4, collate_fn=train_data.collate_fn)
        embed_lst = []
        txt_lst = []
        cnt = 0
        for data_batch in data_loader:
            txts, bows = data_batch
            embed = self.inference_by_bow(bows)
            embed_lst.append(embed)
            txt_lst.append(txts)
            cnt += embed.shape[0]
            if cnt>=num:
                break
        embed_lst = np.concatenate(embed_lst,axis=0)[:num]
        txt_lst = np.concatenate(txt_lst,axis=0)[:num]
        return txt_lst, embed_lst


    def show_topic_word_dist(self, device, topic_id=None):
        topic_word_dist = []
        idxes = torch.eye(self.num_topics).to(device)
        word_dist = self.prop_net.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        if topic_id is not None:
            return word_dist[topic_id]
        return word_dist

    def get_candidates_by_topic_dist(self, topic_dist, device, topK=30):
        '''
        :param topic_dist(torch.Tensor): N * num_topics
        '''
        topic_word_dist = self.show_topic_word_dist(device)
        word_dist = topic_dist.mm(topic_word_dist)
        vals, indices = word_dist.topk(topK, dim=1)
        return indices


    def show_topic_words(self, id2token, device, topic_id=None,topK=5, dictionary=None):
        topic_words = []
        idxes = torch.eye(self.num_topics).to(device)
        word_dist = self.prop_net.decode(idxes)
        word_dist = torch.softmax(word_dist,dim=1)
        vals,indices = torch.topk(word_dist,topK,dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        if id2token==None and dictionary!=None:
            id2token = {v:k for k,v in dictionary.token2id.items()}
        if topic_id==None:
            for i in range(self.num_topics):
                topic_words.append([id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([id2token[idx] for idx in indices[topic_id]])
        return topic_words


        


class PredictHead(nn.Module):

    def __init__(self, feature_in, num_classes):
        super().__init__()
        #self.fc = nn.Linear(feature_in, num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(feature_in, feature_in//2),
            nn.ReLU(),
            nn.Linear(feature_in//2, feature_in//4),
            nn.ReLU(),
            nn.Linear(feature_in//4, num_classes)
        )
        
    def forward(self, x):
        out = self.mlp(x).softmax(1)
        return out

    def loss_ce(self, pred, target, weight=None):
        return F.cross_entropy(pred, target, weight)

    def predict(self, x):
        pass


class ParamPredictor(nn.Module):

    def __init__(self, embd_size, param_types):
        super().__init__()
        self.predictor = nn.Sequential(
                nn.Linear(embd_size, embd_size//2),
                nn.Linear(embd_size//2, embd_size//4),
                nn.Linear(embd_size//4, param_types)
        )
    

    def forward(self, batch):
        return self.predictor(batch)

    def _sample_params(self, target):
        counts = [0, 0, 0, 0]
        for i in range(4):
            counts[i] = torch.sum(target==i).item()
        m = min(counts)
        
        if m == 0:
            return target
        _weights = torch.tensor([(i-m)/i for i in counts]).to(target.device)
        preserve_weight = 1 - _weights
        _weights = F.embedding(target, _weights)
        
        drop_rate = torch.rand(target.shape).to(target.device)
        return (drop_rate > _weights).to(torch.long), preserve_weight

    



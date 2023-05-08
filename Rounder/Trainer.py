import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_sequence,
    pad_packed_sequence,
    pack_sequence
)
from sklearn.metrics.cluster import rand_score
from collections import Counter
from config.bk_config import get_param_type, hash_param
from model.vae import VAE, VAEBase, StateNet
from model.Siamese import SiameseNet
import os
from model.NTM import NTM, PredictHead
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import tqdm
from functools import partial, reduce
from dataset.dataset import Data


#deal with pytorch tf conflict






class Trainer():

    def __init__(self, model: NTM, train_dataset, eval_dataset=None, task=None) -> None:
        self.task = task
        self.model = model
        self.train_dataset: Data = train_dataset
        self.eval_dataset = eval_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, batch_size):
        raise NotImplementedError(f'method train should be implemented manually.')

    def eval(self):
        raise NotImplementedError(f'method eval should be implememted manually.')

    def rand_score(self, **kwargs):
        gold = kwargs.get('gold')
        prediction = kwargs.get('prediction')
        if gold is not None and prediction is not None:
            return rand_score(gold, prediction)
        elif gold is None:
            gold = reduce(lambda x, y: x+y, [i['predicates'] for i in self.train_dataset])
        else:
            raise NotImplementedError()
        # gold = reduce(lambda x, y: x+y, [i['predicates'] for i in self.train_dataset])
        

    def weight_sum(self, logits, weights):
        total_weight = 0
        ret = None
        for w, val in zip(weights, logits):
            assert w >= 0
            ret = w*val if ret is None else ret + w*val
            total_weight += w
        # return ret / total_weight
        return ret

    def gather_loss1(self, loss_dict: dict, weights, loss_ignored_list=[]):
        logits, _weights = [], []
        idx = -1
        for k, v in loss_dict.items():
            idx += 1
            if k in loss_ignored_list:
                continue
            logits.append(v)
            _weights.append(weights[idx])
        return self.weight_sum(logits, _weights)

    def gather_loss(self, loss_dict: dict, loss_weight_dict: dict, loss_ignored_list=[]):
        logits, _weights = [], []
        idx = -1
        for k, v in loss_dict.items():
            if loss_weight_dict.get(k) and not loss_weight_dict[k] == 0:
                logits.append(v)
                _weights.append(loss_weight_dict[k])
        return self.weight_sum(logits, _weights)



class NTMTrainer(Trainer):

    def __init__(self, model: NTM, train_dataset, eval_dataset=None, task=None, predict_param=True, **kwargs) -> None:
        super().__init__(model, train_dataset, eval_dataset=eval_dataset, task=task)
        # self.train_dataset.data = self.train_dataset.data[:1000]
        

        '''if predict_param:
            self.param_predictor = nn.Sequential(
                nn.Linear(self.model.embd_size, self.model.embd_size//2),
                nn.Linear(self.model.embd_size//2, self.model.embd_size//4),
                nn.Linear(self.model.embd_size//4, 4)
            )
            self.param_predictor.to(self.device)
        else:
            self.param_predictor = None'''

    def get_predictor(self, feature_in, num_classes):
        return PredictHead(feature_in, num_classes)
        

    def train(self, batch_size, lr, use_label, use_param_type, loss_scheduler, num_epoch=50, log_steps=1, save_steps=50, **kwargs):
        
        f_lr = kwargs.get('f_lr', lr)
        log_file = kwargs.get('log_file')
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')
        logger = logging.getLogger(__name__)

        if use_label:#weight
            weight = [0] * len(self.train_dataset.dictionary.token2id)
            exists = [0] * len(self.train_dataset.dictionary.token2id)
            for item in self.train_dataset.dataset.values():
                idx = self.train_dataset.dictionary.token2id[item.predicate]
                exists[idx] = 1
                weight[idx] += 1
            weight = torch.tensor(weight) + 0.01
            weight = 1/weight
            weight = weight * torch.tensor(exists)
            weight = weight.to(self.device)
            

        if use_label:
            collate_fn = self.collate_fn_for_label 
            batch_size = batch_size * 2
            train_loader = DataLoader(
                list(self.train_dataset.dataset.keys()),
                batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True
            )
            
            r_lr = f_lr
        else:
            collate_fn = self.get_collate_fn(key='raw', use_label=use_label)

            train_loader = DataLoader(
                self.train_dataset,
                batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True
            )
            r_lr = lr


        tb_writer = SummaryWriter()
        if use_label:
            param_group = [
                {'params': self.model.prop_net.parameters(), 'lr': 1e-7},
                {'params': self.model.predict_head.parameters()},
                {'params': self.model.param_sorter.parameters()}
            ]
        else:
            param_group = [
                {'params': self.model.parameters()}
            ]
        optimizer = Adam(param_group, lr=r_lr)
        
        global_step = 0
        decay = 10
        for current_epoch in range(num_epoch):
            
            current_step = 0

            gold_predicates = []
            topic_predictions = []
            predicate_predictions = []
            
            target_total = []
            inp_total = []
            p_target_total = []
            p_inp_total = []

            for x, gold_pred, param_gold, param_inp in train_loader:
                self.model.train()

                
                #x, bow_y, type_label, sort_label = x
                x, bow_y, label = x
                type_label = torch.tensor(self._param_type2label(self.hash_param(param_inp.tolist())))
                sort_label = param_gold == param_inp
                sort_label[:, 0] = (~sort_label[:, 0])
                bow_y = bow_y.to(self.device)
                x = x.to(self.device)
                param_inp = param_inp.to(self.device)
                param_gold = param_gold.to(self.device)
                sort_label = sort_label.to(self.device).to(torch.long)
                if label is not None:
                    label = label.to(self.device)


                self.model.to(self.device)

                self.model.train()
                optimizer.zero_grad()
                result = self.model(x)
                
                loss_dict = self.get_losses(
                    prop_bow=x.data, 
                    prop_bow_y=bow_y,
                    prop_mu=result['others'][0],
                    prop_sigma=result['others'][1],
                    prop_recon=result['prop_recon'],
                    prop_dist=result['prop_dist'],
                )



                if use_label:
                    label_out = self.model.predict_head_forward(x)
                    label_loss = self.model.label_predict_loss(label_out, label, weight)
                    loss_dict['label_predict'] = label_loss
                    target = label.detach().clone().cpu().tolist()
                    inp = label_out.topk(1)[1].squeeze().detach().clone().cpu().tolist()
                    target_total += target
                    inp_total += inp

                    param_out = self.model.param_sort_forward(param_inp)
                    param_out_loss = self.model.param_sorter.loss(param_out, sort_label)
                    loss_dict['param_sort'] = param_out_loss
                    p_target_total += param_gold.tolist()
                    p_inp_total += param_inp.tolist()
                    
                    
                    
                # record
                topic_predictions += reduce(lambda x, y: x+y, result['prop_dist'].topk(1)[1].tolist())
                predicate_predictions += reduce(lambda x, y: x+y, result['prop_recon'].topk(1)[1].tolist())
                gold_predicates += gold_pred

                # loss_dict = result['loss']
                
                    

                loss_weight_dict = loss_scheduler.loss_weight

                loss = self.gather_loss(loss_dict, loss_weight_dict)
                loss_dict['total_loss'] = loss
                # loss = loss_dict['prop_recon'] + loss_dict['prop_KLD']

                loss.backward()
                optimizer.step()
                loss_scheduler.step()

                if (current_step + 1) % log_steps == 0:
                    
                    log_loss_dict = {}
                    for k, v in loss_dict.items():
                        if not loss_weight_dict[k] == 0:
                        # if k not in loss_ignored_list:
                            log_loss_dict[k] = v
                    tb_writer.add_scalars('loss', log_loss_dict, global_step)
                    t_rs = rand_score(gold_predicates, topic_predictions)
                    p_rs = rand_score(gold_predicates, predicate_predictions)
                    logger.info(f'epoch: {current_epoch}, step: {current_step}, t_rs: {t_rs:0.4f}, p_rs: {p_rs:0.4f}, {[(i, round(j.item(), 5)) for i, j in log_loss_dict.items()]}')
                    if use_label:
                            self.label_log(inp, target, logger)
            
            

                current_step += 1
                global_step += 1
            if (current_epoch+1)/1 == 0:
                decay += 3
            tb_writer.add_scalars(
                'rand_score', 
                {
                    't_rs': rand_score(gold_predicates, topic_predictions), 
                    'p_rs': rand_score(gold_predicates, predicate_predictions)
                },
                current_epoch
            )

            topic_words = self.model.show_topic_words(self.train_dataset.dictionary.id2token, self.device)
            topic_count = Counter(topic_predictions)
            tb_writer.add_embedding(self.model.prop_net.topic_embd.weight, tag='topic_embd', global_step=current_epoch)
            word_label = [self.train_dataset.dictionary.id2token[i] for i in range(len(self.train_dataset.dictionary.id2token.values()))]
            tb_writer.add_embedding(self.model.prop_net.word_embd.weight,metadata=word_label, tag='word_embd', global_step=current_epoch)
            # tb_writer.add_text('topic', f'{}')
            for idx, item in enumerate(topic_words):
                tb_writer.add_text(f'topic{idx}', f'{topic_count[idx]}\t {item}', current_epoch)
                print(f'{topic_count[idx]}\t {item}')
            logger.info(f'epoch: {current_epoch}, t_rs: {rand_score(gold_predicates, topic_predictions)}, p_rs: {rand_score(gold_predicates, predicate_predictions)}')
            if use_label:
                self.label_log(inp_total, target_total, logger)
        tb_writer.close()
        
        return {
            't_rs': rand_score(gold_predicates, topic_predictions),
            'p_rs': rand_score(gold_predicates, predicate_predictions)
        }


    def collate_fn(self, data, key='raw', use_label=False):
        '''
        :param mapping: token to class index
        '''
        key2x = {
            'raw': 'sentences',
            'pre': 'sentences_pre',
            'post': 'sentences_post',
            'pre_post': 'sentences_pre_post'
        }
        if not key2x.get(key):
            raise NotImplementedError()
        
        xs = reduce(lambda x, y: x+y, [i[key2x[key]] for i in data])

        xs_pad = [torch.tensor(i) for i in xs]
        xs_pad = pad_sequence(xs_pad, batch_first=True, padding_value=self.train_dataset.pad_id)
        xs_length = [len(i) for i in xs]
        xs_pad_pack = pack_padded_sequence(xs_pad, xs_length, batch_first=True, enforce_sorted=False)


        if use_label:
            labels = reduce(lambda x, y: x+y, [i['psuedo_label'] for i in data])
            #labels = [self.train_dataset.dictionary.id2token[i[0]] for i in labels]
            labels = torch.tensor(labels).squeeze()
        else:
            labels = None
        


        bow = [self.train_dataset.bow(i, return_vec=True) for i in xs]
        bow = torch.tensor(bow)


        state = [reduce(lambda x, y: x+y, i['predicates']) for i in data]
        gold_state = reduce(lambda x, y: x+y, state)
        
        
        # params
        params = [i['psuedo_params'] for i in data]
        extract_params = [i['extract_params'] for i in data]
        params_flat = reduce(lambda x, y: x+y, params)
        extract_params_flat = reduce(lambda x, y: x+y, extract_params)
        params_pad = pad_sequence([torch.tensor(i) for i in params_flat], batch_first=True)
        extract_params_pad = pad_sequence([torch.tensor(i) for i in extract_params_flat], batch_first=True)


        #return (xs_pad_pack, bow, labels), gold_state, params_type, params_sort_label
        return (xs_pad_pack, bow, labels), gold_state, params_pad, extract_params_pad # x, gold, param_gold, param_inp
        

    def get_collate_fn(self, key, use_label):
        return partial(self.collate_fn, key=key, use_label=use_label)

    def save_model(self, save_path):
        dir, _ = os.path.split(save_path)
        if dir:
            os.makedirs(dir, exist_ok=True)
        torch.save(self.model, save_path)


    def get_losses(self, **kwargs):
        prop_bow = kwargs.get('prop_bow')
        prop_bow_y = kwargs.get('prop_bow_y')
        #state_bow = kwargs.get('state_bow')
        #label_action = kwargs.get('label_action')

        prop_mu = kwargs.get('prop_mu')
        prop_sigma = kwargs.get('prop_sigma')
        #state_mu = kwargs.get('state_mu')
        #state_sigma = kwargs.get('state_sigma')
        prop_recon = kwargs.get('prop_recon')
        #prop_dist = kwargs.get('prop_dist')
        #state_dist = kwargs.get('state_dist')
        #tate_recon = kwargs.get('state_recon')
        #action_prediction = kwargs.get('action_prediction')

        loss_prop_KLD = self.loss_KLD(prop_mu, prop_sigma)
        loss_prop_recon = self.loss_recon(prop_bow_y if prop_bow_y is not None else prop_bow, prop_recon)
        #loss_state_KLD = self.loss_KLD(state_mu, state_sigma)
        #loss_state_recon = self.loss_recon(state_bow, state_recon)
        #loss_state_ce = F.cross_entropy(state_dist, label_action)
        #loss_action_ce = F.cross_entropy(action_prediction, label_action)
        
        

        return {
            'prop_recon': loss_prop_recon,
            'prop_KLD': loss_prop_KLD,
            #'state_recon': loss_state_recon,
            #'state_KLD': loss_state_KLD,
            #'state_ce': loss_state_ce,
            #'action_ce': loss_action_ce
            
        }



        

    def loss_KLD(self, mu, log_sigma):
        return self.model.prop_net.loss_KLD(mu, log_sigma)

    def loss_recon(self, x, x_recon):
        return self.model.prop_net.loss_reconstruct(x, x_recon)


    def _make_siam_pack(self, state, word_dist, *args):
        _pad = pad_sequence([torch.tensor(i) for i in state], batch_first=True).to(word_dist.device)
        _pad = self.model.prop_net.word_embd(_pad)
        #_pad = F.embedding(_pad, self.model.prop_net.word_embd.weight, None, None, 2., False, False)
        pack_y = pack_padded_sequence(_pad, [len(i) for i in state], True, False)
        pack_x = self.model._make_state_net_input(torch.matmul(word_dist, self.model.prop_net.word_embd.weight), *args)
        return pack_x, pack_y


    def _siam_process(self, state, word_dist, *args):
        pack_x, pack_y = self._make_siam_pack(state, word_dist, *args)
        hid_x, hid_y = self.siamese_net(pack_x, pack_y)
        return hid_x, hid_y

    def predict_param_process(self, topic_dist, target):
        out = torch.matmul(topic_dist, self.model.prop_net.topic_embd.weight)
        out = self.param_predictor(out)
        drop, class_weights = self._sample_params(target)
        '''
        :param out(torch.Tensor): (N*C)
        :param target(torch.Tensor): (N), target_i \in {0, 1, ... , C - 1}
        :param class_weights: 与target中每个类别出现的次数成反比，目的是提高出现次数较少的类别的权重
        N: Batch size, C: 参数的类别个数
        '''
        return F.cross_entropy(out, target, class_weights)
        

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



    def _param_type2label(self, param):
        

        if isinstance(param, int):
            return param

        if isinstance(param, list):
            if isinstance(param[0], int):
                return {1: 0, 2: 1, 5: 2, 6: 3}[sum(param)]
            elif isinstance(param[0], list):
                return [self._param_type2label(i) for i in param]

        else:
            raise ValueError()

        

    def hash_param(self, param):
        if isinstance(param, int):
            if param == 0:
                return 0
            token = self.train_dataset.dictionary.id2token[param]
            token_t = self.train_dataset.args_type[token]
            return {'type0': 1, 'type1': 5}[token_t]
        elif isinstance(param, list):
            return [self.hash_param(i) for i in param]
        else:
            raise ValueError()


    def predict_label_process(self, x, type_label):
        hid = self.model.prop_net.get_hid(x)
        out = self.label_predictor(hid)
        return out, F.cross_entropy(out, type_label)

    def label_log(self, inp, target, logger):
        acc = accuracy_score(target, inp)
        prec = precision_score(target, inp, average='micro')
        rec = recall_score(target, inp, average='micro')
        f1 = f1_score(target, inp, average='micro')
        logger.info(f'acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}')

    def collate_fn_for_label(self, data):
        # data is (plan_id, state_id, pred_id) list
        sents = [self.train_dataset.sents[i] for i in data]
        xs = self.train_dataset.batch_encode_sentence(sents)
        xs_pad = pad_sequence([torch.tensor(i) for i in xs], batch_first=True)
        xs_length = [len(i) for i in xs]
        xs_pad_pack = pack_padded_sequence(xs_pad, xs_length, batch_first=True, enforce_sorted=False)

        bow = [self.train_dataset.bow(i, return_vec=True) for i in xs]
        bow = torch.tensor(bow)

        gold_state = [self.train_dataset.gold[i].predicate for i in data]

        psuedo_item = [self.train_dataset.dataset[i] for i in data]
        labels = [i.predicate for i in psuedo_item]
        labels = self.train_dataset.encode_sentence(' '.join(labels))
        
        params = [i.params for i in psuedo_item]
        params = [reduce(lambda x, y: x+y, self.train_dataset.batch_encode_sentence(i)) for i in params]
        extract_params = [self.train_dataset.extract[i] for i in data]
        extract_params = [reduce(lambda x, y: x+y, self.train_dataset.batch_encode_sentence(i)) for i in extract_params]
        
        params_pad = pad_sequence([torch.tensor(i) for i in params], batch_first=True)
        extract_params_pad = pad_sequence([torch.tensor(i) for i in extract_params], batch_first=True)

        return (xs_pad_pack, bow, torch.tensor(labels)), gold_state, params_pad, extract_params_pad



class McTrainer(NTMTrainer):

    def hash_param(self, param):
        if isinstance(param, int):
            if param == 0:
                return 0
            token = self.train_dataset.dictionary.id2token[param]
            token_t = self.train_dataset.args_type[token]
            return {'type0': 1, 'type1': 3, 'type2': 5}[token_t]
        elif isinstance(param, list):
            return [self.hash_param(i) for i in param]
        else:
            raise ValueError()

    def _param_type2label(self, param):
        

        if isinstance(param, int):
            return param

        if isinstance(param, list):
            if isinstance(param[0], int):
                return {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[sum(param)]
            elif isinstance(param[0], list):
                return [self._param_type2label(i) for i in param]

        else:
            raise ValueError()


class BkTrainer(NTMTrainer):
    def hash_param(self, param):
        return hash_param(param, self.train_dataset.dictionary.id2token)

    def _param_type2label(self, param):
        return get_param_type(param)

from collections import defaultdict
from functools import partial, reduce
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
#from trainer import predict, train_state_model, train_ntm
from model.models import MappingModel, StateModel, ModelOutputForPlan
from model import Lda
from model.dataset.data_reader import SimpleData, StateData
from dataset.dataset import Data, PsuedoItem
from dataset.tokenizer import BaseTokenizer
import torch
import config.nb_config, config.mc_config
import os
from utils import check_task
#from config.mc_config import model_config, training_config, args_type, label2word
import argparse
import nltk
import pickle as pkl
from .Trainer import McTrainer, NTMTrainer, BkTrainer


class Predictor(NTMTrainer):
    def __init__(self, model, train_dataset, eval_dataset=None, stop_words=None):
        self.model = model
        self.train_dataset: Data = train_dataset
        self.eval_dataset = eval_dataset
        self.stop_words = stop_words if stop_words else []
        for idx, item in enumerate(self.stop_words):
            if isinstance(item, str):
                self.stop_words[idx] = self.train_dataset.dictionary.token2id[item]



    def predict(self):
        raise NotImplementedError()


class NTMPredictor(Predictor):

    def train(self):
        raise NotImplementedError()

    def predict(self, batch_size, train_out_file, eval_out_file, collate_key='pre'):
        '''
        predict both train and eval dataset
        '''
        train_predictions = self.predict_train(batch_size, train_out_file, collate_key)
        eval_predictions = self.predict_eval(batch_size, eval_out_file, collate_key)

        return train_predictions, eval_predictions


    def predict_train(self, batch_size, out_file, collate_key='raw', **kwargs):
        return self.predict_dataset(self.train_dataset, batch_size, out_file, collate_key, **kwargs)

    def predict_eval(self, batch_size, out_file, collate_key='pre', **kwargs):
        return self.predict_dataset(self.eval_dataset, batch_size, out_file, collate_key, **kwargs)

    def predict_by_head(self, dataset: Data, batch_size, mapping, collate_key='pre'):
        result = []
        '''_map = {}
        for key, value in mapping.items():
            _map[value] = key'''
        collate_fn = self.get_collate_fn(collate_key, None)
        loader = DataLoader(
            dataset, 
            batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        label2type = {}

        with torch.no_grad():
            for item in loader:
                
                inp, param, ids, param_inp = item
                param_type = self._param_type2label(self.hash_param(param))
                out = self.model.predict_head_predict(inp)
                predictions = out.topk(len(dataset.dictionary.id2token))[1].detach().clone().cpu().tolist()
                stop_words = self.get_stop_words(True)
                params_resort = self.model.param_sort_predict(param_inp).detach().clone().cpu().tolist()
                #for id, pr, pa, pt in zip(ids, predictions, param, param_type):
                for id, pr, pa, pt in zip(ids, predictions, params_resort, param_type):
                    #pr = _map[pr]
                    _pr = None
                    for candidate in pr:
                        if candidate in stop_words:
                            continue
                        if label2type.get(candidate) is None or label2type.get(candidate) == pt:
                            _pr = candidate
                            label2type[candidate] = pt
                            break
                        
                    _pr = self.train_dataset.dictionary.id2token[_pr]
                    assert _pr
                    _pa = self.train_dataset.decode_sentence(pa, True).split(' ')

                    
                    

                    

                    data = {
                        'id': id,
                        'predicate': _pr,
                        'extract_params': _pa,
                        'gold_params': _pa,
                        'label': None,
                        'sentence': None,
                        'is_goal': False,
                        'args_type': dataset.args_type
                    }
                    result.append(ModelOutputForPlan(data))
                    plan_id, state_id, pred_id = id
                    dataset.prediction[(plan_id, state_id, pred_id)] = PsuedoItem(_pr, _pa)
            result.sort(key=lambda x: x.id)
        return result
        
    def predict_by_topic(self, dataset, batch_size, out_file, collate_key='raw'):
        result = []
        collate_fn = self.get_collate_fn(collate_key, None)

        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        topic_word_ids = self.model.show_topic_word_ids(45)
        topic2label = []
        label2args = {}
        for _ in topic_word_ids:
            topic2label.append({})
        stop_words = self.get_stop_words(True)
        # 参数不同则选择下一个单词作为label
        with torch.no_grad():
            for item in loader:
                self.model.eval()
                inp, param, ids, _ = item
                topic_ids = self.model.predict(inp, stop_words=self.get_stop_words(True))
                
                
                param_type = self._param_type2label(self.hash_param(param))
                for index, topic, pa, id in zip(range(len(ids)), topic_ids, param, ids):
                    _pt = param_type[index]
                    pa = self.resort_param(pa, self.train_dataset.args_type, self.train_dataset.dictionary.id2token)
                    pr = None

                                
                        
                    pr = topic2label[topic].get(_pt)
                    if not pr:
                        for word_idx in topic_word_ids[topic]:
                            if not word_idx in stop_words:
                                stop_words.append(word_idx)
                                topic2label[topic][_pt] = word_idx
                                pr = word_idx
                                break
                    if not pr:
                        pr = self.get_predicate_by_topic(topic, topic_word_ids, self.get_stop_words(True))
                    _pr = self.train_dataset.decode_sentence([pr], True)
                    assert _pr
                    _pa = self.train_dataset.decode_sentence(pa, True).split(' ')

                    data = {
                        'id': id,
                        'predicate': _pr,
                        'extract_params': _pa,
                        'gold_params': _pa,
                        'label': None,
                        'sentence': None,
                        'is_goal': False,
                        'args_type': self.train_dataset.args_type
                    }
                    result.append(ModelOutputForPlan(data))
                    plan_id, state_id, pred_id = id
                    dataset.prediction[(plan_id, state_id, pred_id)] = PsuedoItem(_pr, _pa)
            result.sort(key=lambda x: x.id)
        return result

    def predict_dataset(self, dataset, batch_size, out_file, collate_key='raw', **kwargs):
        by_head = kwargs.get('by_head', False)
        if by_head:
            mapping = kwargs.get('mapping')
            result = self.predict_by_head(dataset, batch_size, None, collate_key)
        else:
            result = self.predict_by_topic(dataset, batch_size, collate_key)
        os.makedirs(os.path.split(out_file)[0], exist_ok=True)
        with open(out_file, 'wb') as f:
            pkl.dump(result, f)
        return result
        

    def collate_fn(self, data, key='raw', use_label=False):
        params = [i['psuedo_params'] for i in data]
        params = reduce(lambda x, y: x+y, params)

        ids = [
            [
                [i['plan_idx'], i['trace_idx'], j] for j in range(len(i['sentences']))
            ] for i in data
        ]
        ids = reduce(lambda x, y: x+y, ids)
        su = super().collate_fn(data, key=key, use_label=use_label)
        return su[0][0], params, ids, su[3]



    def get_stop_words(self, return_id=False):
        ret = self.train_dataset.get_special_tokens(return_id)
        for token in self.train_dataset.args_type.keys():
            if return_id:
                if self.train_dataset.dictionary.token2id.get(token):
                    ret.append(self.train_dataset.dictionary.token2id[token])
            else:
                ret.append(token)
        return list(set(ret+self.stop_words))

    def get_predicate_by_topic(self, topic, topic_word_ids, stop_words):
        
        #topic_word_ids = self.model.show_topic_word_ids(45)
        for item in topic_word_ids[topic]:
            if not item in stop_words:
                return item
        return None

    def resort_param(self, param, args_type, id2token):
        pa = param[:]
        pt = self._param_type2label(self.hash_param(param))
        if pt == 3:
            param_token = id2token[pa[0]]
            if args_type[param_token] == 'type1':
                pa = [pa[1], pa[0]]

        return pa


class McPredictor(McTrainer, NTMPredictor):

    def resort_param(self, param, args_type, id2token):
        pa = param[:]
        pt = self._param_type2label(self.hash_param(param))
        if pt == 3 or pt == 5:
            param_token = id2token[pa[1]]
            if args_type[param_token] == 'type0':
                pa = [pa[1], pa[0]]
        return pa
    

class BkPredictor(BkTrainer, NTMPredictor):
    def resort_param(self, param, args_type, id2token):
        pa = param[:]
        pt = self._param_type2label(self.hash_param(param))
        if pt == 5:
            param_token = id2token[pa[1]]
            if args_type[param_token] == 'type3':
                pa = [pa[1], pa[0]]
        return pa



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_file', type=str, help='model file.', default='model_round1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--prediction_out_file', type=str, default='prediction.pkl')
    #parser.add_argument('--from_lda', action='store_true', default=False)


    parser.add_argument('--task', type=str, default='nb')
    parser.add_argument('--cache_dir', type=str, default='tmp')
    parser.add_argument('--data_file', type=str, default='data/Newblocks/nb_train.pkl')
    parser.add_argument('--dataset_from_scratch', action='store_true')
    parser.add_argument('--dataset_binary_file', type=str, default='tmp/nb/dataset/dataset_round0')

    args = parser.parse_args()

    os.makedirs(os.path.join('tmp', args.task, 'predictions'), exist_ok=True)

    args.model_file = os.path.join('tmp', args.task, 'model', args.model_file)
    args.prediction_out_file = os.path.join('tmp', args.task, 'predictions', args.prediction_out_file)
    if not args.task+'_' in args.data_file:
        raise ValueError(f'data file {args.data_file} not suitable for task {args.task}')

    TASK_CONFIG = {
        'nb': config.nb_config, 
        'mc': config.mc_config
    }
    CFG = TASK_CONFIG[args.task]


    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    check_task(args.data_file, args.task)

    cache_dir = os.path.join(args.cache_dir, args.task)
    os.makedirs(cache_dir, exist_ok=True)
    # args.args_type = args_type
    #args.dataset = dataset
    if args.dataset_from_scratch and args.data_file:
        dataset = Data(
            args.data_file, 
            cache_dir, 
            args_type=CFG.args_type, 
            force=False,
            tokenizer=BaseTokenizer()
        )
        dataset_save_dir, _ = os.path.split(args.dataset_binary_file)
        os.makedirs(dataset_save_dir, exist_ok=True)
        dataset.save(args.dataset_binary_file)
        
    elif args.dataset_binary_file:
        dataset = Data.load(args.dataset_binary_file)


    model = torch.load(args.model_file)
    model.to(args.device)

    predictor = NTMPredictor(model, dataset)
    predictor.predict(args.batch_size, args.prediction_out_file, 'pre')







from copy import copy, deepcopy

from typing import Dict
from sklearn.metrics import rand_score

from numpy.lib.arraysetops import isin
from plan_generator.Plan import Predicate, State

from plan_generator.generator import parse_result
import numpy as np
import os
import nltk
import torch
import pickle as pkl
import logging
from torch.nn.modules.module import T
from torch.nn.modules.transformer import TransformerEncoder
from torch.utils import data
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from collections import Counter, defaultdict, namedtuple
import tqdm
from gensim.corpora import Dictionary
from functools import reduce
import glob
from .tokenizer import BaseTokenizer, Tokenizer



class PsuedoItem():
    def __init__(self, predicate, params) -> None:
        self.__predicate = predicate
        self.__params = params

    @property
    def predicate(self):
        return deepcopy(self.__predicate)
    
    @property
    def params(self):
        return self.__params

    def __repr__(self):
        return f'PsuedoItem({self.__predicate}, {self.__params})'

class Data(Dataset):
    def __init__(self, file_in, cache_dir='tmp/nb', args_type=None, force=False, tokenizer=None, limits=-1) -> None:
        super().__init__()

        _, file_in_suffix = os.path.split(file_in)
        os.makedirs(cache_dir, exist_ok=True)

        self.dictionary = Dictionary()
        self.tokenizer: Tokenizer = tokenizer
        self.data = []

        self.psuedo = {} # made dataset
        self.sents = {} # raw sents
        self.gold = {} # gold predicate and params
        self.prediction = {} # prediction of model(predicate params)
        self.extract = {} # extract_params
        
        self.dataset = {}
        self.propsitions = []
        self.actions = []
        self.sentences = []
        self.args_type = args_type
        self.args_tokens = list(args_type.keys()) if args_type is not None else None
        self.pad_token, self.pad_id = '<PAD>', 0

        _special_tokens = {
            self.pad_token: self.pad_id, 
        }
        if self.args_type:
            _special_token_idx = 1
            _type_count = defaultdict(int)
            for arg, t in self.args_type.items():
                _special_tokens[f'<{t}_arg{_type_count[t]}>'] = _special_token_idx
                _type_count[t] += 1
                _special_token_idx += 1
                
        
        
        with open(file_in, 'rb') as f:
            self.raw = pkl.load(f)
            if limits != -1:
                limits = min(len(self.raw), limits)
                self.raw = self.raw[:limits]

        for data in self.raw:
            # self.labels

            plan_id = data['id']
            goal_state = data['goal_state']
            data['state'][0] = data['initial_state']
            _plan = [None] + data['plan'][:-1] + [None]
            for trace_idx, t, s in zip(range(len(data['text_trace'])), data['text_trace'], data['state']):
                t = self._pre_process_raw_text(t)
                
                _tokens = self.tokenizer.tokenize(t)
                assert '' not in t
                self.dictionary.add_documents(_tokens)
                self.sentences += t
                is_last_state = True if trace_idx == len(data['text_trace']) - 1 else False
                sents = []
                predictions = []
                labels = []
                psuedo_labels = []
                psuedo_params = []
                gold_param_list = []
                extract_param_list = []
                for predicate_idx, sent, pred in zip(range(len(t)), t, s):
                    is_goal = True if is_last_state and pred in goal_state else False
                    pred = pred.replace('(', '')
                    pred = pred.replace(')', '')
                    gold_params = pred.split(' ')[1: ]
                    pred = pred.split(' ')[0]
                    extract_params = self.extact_params_from_sentence(sent)
                    sents.append(sent)
                    labels.append(pred)
                    psuedo_labels.append(sent)
                    
                    psuedo_params.append(extract_params)

                    self.psuedo[(plan_id, trace_idx, predicate_idx)] = PsuedoItem(sent, extract_params)
                    self.prediction[(plan_id, trace_idx, predicate_idx)] = PsuedoItem(sent, extract_params)
                    self.sents[(plan_id, trace_idx, predicate_idx)] = sent
                    self.gold[(plan_id, trace_idx, predicate_idx)] = PsuedoItem(pred, gold_params)
                    self.extract[(plan_id, trace_idx, predicate_idx)] = extract_params

                    gold_param_list.append(gold_params)
                    extract_param_list.append(extract_params)
                    local_args_type = {}
                    pre_args_type = {}
                    post_args_type = {}
                    pre_post = [_plan[trace_idx + 1], _plan[trace_idx]]
                    for pre_post_idx, _ in enumerate(pre_post):
                        if _:
                            pre_post[pre_post_idx] = pre_post[pre_post_idx].replace('(', '')
                            pre_post[pre_post_idx] = pre_post[pre_post_idx].replace(')\n', '')
                            pre_post[pre_post_idx] = pre_post[pre_post_idx].split(' ')

                            self.actions.append(pre_post[pre_post_idx][0])
                            self.propsitions += labels
                            

                    pre_of, post_of = pre_post
                    if self.args_type:
                        if pre_of:
                            for _arg in pre_of[1: ]:
                                pre_args_type[_arg] = self.args_type[_arg]
                                local_args_type[_arg] = self.args_type[_arg]

                        if post_of:
                            for _arg in post_of[1: ]:
                                post_args_type[_arg] = self.args_type[_arg]
                                local_args_type[_arg] = self.args_type[_arg]
                    else:
                        pre_args_type, post_args_type, local_args_type = None, None, None
                
                self.data.append(
                    {
                        'plan_idx': plan_id,
                        'trace_idx': trace_idx,
                        'gold_params_list': gold_param_list,
                        'extract_params_list': extract_param_list,
                        'is_goal': is_goal,
                        'sentences': sents,
                        'formatted_sentences': [self.format_sentence_with_args(i, local_args_type) for i in sents],
                        'formatted_sentences_with_pre': [self.format_sentence_with_args(i, pre_args_type) for i in sents],
                        'formatted_sentences_with_post': [self.format_sentence_with_args(i, post_args_type) for i in sents],
                        'gold_predicates': labels,
                        'predicates': labels,
                        # 'predicates': [list(self.dictionary.token2id.keys())[1] for _ in labels], # 初始化的dataset没有标签，也不会用来训练，所以随便给一个
                        'pre_of': _plan[trace_idx + 1],
                        'post_of': _plan[trace_idx],
                        'psuedo_label': [self.format_sentence_with_args(i, pre_args_type) for i in sents],
                        'psuedo_params': psuedo_params
                    }
                )

        self.dictionary.patch_with_special_tokens(_special_tokens)
        for k, v in self.dictionary.token2id.items():
            self.dictionary.id2token[v] = k
        
        self.actions = list(set(self.actions))
        self.actions = ['<PREON>'] + self.actions + ['<POSTON>'] # pre of nothing, post of nothing
        self.propsitions = list(set(self.propsitions))

    def __getitem__(self, index):
        data = self.data[index]
        pre_of = data['pre_of']
        post_of = data['post_of']
        if pre_of:
            pre_of = pre_of.replace('(', '')
            pre_of = pre_of.split(' ')[0]
        else:
            pre_of = '<PREON>'

        if post_of:
            post_of = post_of.replace('(', '')
            post_of = post_of.split(' ')[0]
        else:
            post_of = '<POSTON>'
            
        try:
            predicates = self.batch_encode_sentence(data['predicates'])
        
        except KeyError: #predicates like handempty OOV
            
            #predicates_idx = [self.dictionary.token2id[prop2word[i]] for i in data['predicates']]
            #predicates = [[i] for i in predicates_idx]
            predicates_idx = [self.propsitions.index(i) for i in data['predicates']]
            predicates = [[i+20] for i in predicates_idx]
            psuedo_label, psuedo_params = [], []
            plan_id = data['plan_idx']
            state_id = data['trace_idx']
            for idx, _ in enumerate(data['sentences']):
                item = self.psuedo[(plan_id, state_id, idx)]
                psuedo_label.append(item.predicate)
                psuedo_params.append(item.params)

        return {
            'sentences': self.batch_encode_sentence(data['sentences']),
            'sentences_pre': self.batch_encode_sentence(data['formatted_sentences_with_pre']),
            'sentences_post': self.batch_encode_sentence(data['formatted_sentences_with_post']),
            'sentences_pre_post': self.batch_encode_sentence(data['formatted_sentences']),
            'pre_of': self.actions.index(pre_of) if pre_of else pre_of,
            'post_of': self.actions.index(post_of) if post_of else post_of, 
            'predicates': predicates, 
            'plan_idx': data['plan_idx'],
            'trace_idx': data['trace_idx'],
            'gold_params': self.batch_encode_sentence([' '.join(i) for i in data['gold_params_list']]),
            'extract_params': self.batch_encode_sentence([' '.join(i) for i in data['extract_params_list']]),
            #'psuedo_label': self.batch_encode_sentence(data['psuedo_label']),
            #'psuedo_params': self.batch_encode_sentence([' '.join(i) for i in data['psuedo_params']])
            'psuedo_label': self.batch_encode_sentence(psuedo_label),
            'psuedo_params': self.batch_encode_sentence([' '.join(i) for i in psuedo_params])
        }

    def __len__(self):
        return len(self.data)


    def encode_sentence(self, sent):
        assert isinstance(sent, str)
        tokens = self.tokenizer.tokenize(sent)
        return [self.dictionary.token2id[i] for i in tokens]

    def batch_encode_sentence(self, batch):
        assert isinstance(batch, list)
        assert isinstance(batch[0], str)
        return [self.encode_sentence(i) for i in batch]

    def _pre_process_raw_text(self, text):
        '''
        将原始数据中的一个text（包含多个句子，句子间由. 隔开）
        :param text: 一条文本
        :return: list
        '''
        ret = []
        t = text.split('.')[:-1]
        for sent in t:
            sent = sent.lower()
            if sent[0] == ' ':
                sent = sent[1:]
            ret.append(sent)
        return ret


    def format_sentence_with_args(self, sent: str, args_type: dict = None):
        '''
        substitude args in a sentence with symbols.
        :param args_type(dict): dict maps args to their type, if args_type is None, use self.args_type
        
        '''
        tokens = self.tokenizer.tokenize(sent)
        if args_type is None:
            args_type = self.args_type
        if args_type is None:
            raise ValueError(f'args_type should be a dict')
        arg_count = {}
        for item in list(set(args_type.values())):
            arg_count[item] = 0
        for token_id, token in enumerate(tokens):
            for arg, t in args_type.items():
                if arg == token:
                    tokens[token_id] = f'<{t}_arg{arg_count[t]}>'
                    arg_count[t] += 1
        return ' '.join(tokens)


    def extact_params_from_sentence(self, sent):
        tokens = nltk.word_tokenize(sent)
        params = []
        if self.args_tokens is None:
            return params
        for token in tokens:
            for param in self.args_tokens:
                if token == param:
                    params.append(token)
        return params
  
    def decode_sentence(self, tokens, skip_special=False):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        assert isinstance(tokens, list)
        assert len(tokens) == 0 or isinstance(tokens[0], int)

        if skip_special:
            res = [self.dictionary.id2token[idx] for idx in tokens if idx not in [self.pad_id]]
        else:
            res = [self.dictionary.id2token[idx] for idx in tokens]
        sent = ' '.join(res)
        return sent

    def batch_decode_sentence(self, batch, skip_special=False):
        if isinstance(batch, torch.Tensor):
            batch = batch.tolist()
        assert isinstance(batch, list)
        assert isinstance(batch[0], list)

        '''res = [
            [self.dictionary.id2token[idx] for idx in tokens] for tokens in batch
        ]
        sents = [
            ' '.join(tokens) for tokens in res
        ]'''
        sents = [
            self.decode_sentence(i, skip_special) for i in batch
        ]
        return sents


    def bow(self, doc, return_vec=False):
        '''
        :param doc(str | list(str) | torch.Tensor | list(int) | list(list(int))):
        '''
        one_line = False
        if isinstance(doc, str):
            doc = self.tokenizer.tokenize(doc)
            one_line = True
        elif isinstance(doc, torch.Tensor):
            doc = doc.tolist()
            if isinstance(doc[0], int):
                one_line = True
                doc = self.decode_sentence(doc)
                doc = self.tokenizer.tokenize(doc)
            elif isinstance(doc[0], list):
                doc = self.batch_decode_sentence(doc)
                doc = self.tokenizer.tokenize(doc)

        elif isinstance(doc, list):
            if isinstance(doc[0], str):
                doc = self.tokenizer.tokenize(doc)
            elif isinstance(doc[0], int):
                one_line = True
                doc = self.decode_sentence(doc)
                doc = self.tokenizer.tokenize(doc)
            elif isinstance(doc[0], list):
                assert len(doc[0]) == 0 or isinstance(doc[0][0], int)
                doc = self.batch_decode_sentence(doc)
                doc = self.tokenizer.tokenize(doc)

        bow = self.dictionary.doc2bow(doc) if one_line else [self.dictionary.doc2bow(i) for i in doc]
        if return_vec:
            if one_line:
                ret = [0] * len(self.dictionary.token2id.values())
                for pos, val in bow:
                    ret[pos] = val
            else:
                ret = []
                for item in bow:
                    _ret = [0] * len(self.dictionary.token2id.values())
                    for pos, val in item:
                        _ret[pos] = val
                    ret.append(_ret)
            return ret
        
        return bow

    def get_special_tokens(self, return_id=False):
        ret = []
        for k, v in self.dictionary.token2id.items():
            if '<' in k:
                ret.append(k)
            elif self.args_type and k in self.args_type.keys():
                ret.append(k)
        return [self.dictionary.token2id[k] for k in ret] if return_id else ret

    def save(self, path):
        d, f = os.path.split(path)
        os.makedirs(d, exist_ok=True)
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    def rebuild_data(self, soln_dir, res_dir, specific_res_file=None):
        '''
        如果给定specific_res_file，则用specific_res_file生成plan
        result file命名方式为"Result.out|{id1}|{id2}|{id3}|...|"
        solution file命名方式为"Soln{id}"
        '''

        res_file_list = glob.glob(os.path.join(res_dir, 'Result.out|*'))
        plans = []
        for res_file in res_file_list:
            
            soln_ids = res_file.split('|')[1:-1]
            soln_files = ['Soln' + i for i in soln_ids]
            soln_files = [os.path.join(soln_dir, i) for i in soln_files]
            res_file = specific_res_file if specific_res_file else res_file
            plans += parse_result(res_file, soln_files=soln_files)
        plans.sort(key=lambda x: x.plan_id)
        self.data.sort(key=lambda x: (x['plan_idx'], x['trace_idx']))
        # states = reduce(lambda x, y: x+y, [i.states for i in plans])
        data_idx = 0
        for plan in plans:
            plan_id = plan.plan_id
            plan_id = int(plan_id)
            for state_id, state in enumerate(plan.states):
                # state_in_dataset = dataset.data[data_idx]
                
                assert self.data[data_idx]['trace_idx'] == state_id
                assert self.data[data_idx]['plan_idx'] == plan_id
                self.data[data_idx]['predicates'] = [i.name.lower() for i in state.content]
                assert '' not in self.data[data_idx]['predicates']
                self.data[data_idx]['extract_params_list'] = [i.params for i in state.content]
                data_idx += 1
        # dataset.labels = None
        

    def __setitem__(self, key, value):
        if len(key) == 1:
            plan_id = key
            raise NotImplementedError()
            
        elif len(key) == 2:
            plan_id, state_id = key
            assert isinstance(value, dict)
            assert value['plan_id'] == plan_id and value['state_id'] == state_id
            
            for idx, item in enumerate(self.data):
                Warning('this is a warning.')
                if item['plan_idx'] == plan_id and item['trace_idx'] == state_id:
                    self.data[idx] = value
                
        elif len(key) == 3:
            plan_id, state_id, pred_id = key
            for idx, item in enumerate(self.data):
                if item['plan_idx'] == plan_id and item['trace_idx'] == state_id:
                    if len(item['sentences'])-1 < pred_id:
                        raise ValueError('pred_id out of range.')
                    if isinstance(value, Predicate):
                        value = (value.name.lower(), [i.lower() for i in value.params])
                    if isinstance(value, tuple):
                        assert isinstance(value[0], str)
                        assert isinstance(value[1], list)
                        self.data[idx]['psuedo_label'][pred_id] = value[0]
                        self.data[idx]['psuedo_params'][pred_id] = value[1]
        elif len(key) == 4:
            plan_id, state_id, pred_id, _key = key
            for idx, item in enumerate(self.data):
                if item['plan_idx'] == plan_id and item['trace_idx'] == state_id:
                    if len(item['sentences'])-1 < pred_id:
                        raise ValueError('pred_id out of range.')
                    
                    self.data[idx][_key][pred_id] = value[0]
                    
        else:
            raise NotImplementedError(f'length of index should be (1, 2, 3, 4), got {len(key)}.')
    
    def iloc(self, *args):
        if len(args) == 1:
            plan_id = args[0]
            raise NotImplementedError()
        elif len(args) == 2:
            plan_id, state_id = args
            for item in self.data:
                if item['plan_idx'] == plan_id and item['trace_idx'] == state_id:
                    return deepcopy(item)

        elif len(args) == 3:
            plan_id, state_id, pred_id = args
            for item in self.data:
                if item['plan_idx'] == plan_id and item['trace_idx'] == state_id:
                    return (item['psuedo_label'][pred_id], item['psuedo_params'][pred_id])
        elif len(args) == 4:
            plan_id, state_id, pred_id, key = args
            for item in self.data:
                if item['plan_idx'] == plan_id and item['trace_idx'] == state_id:
                    return item[key][pred_id]

        else:
            raise ValueError()


    @classmethod
    def load(self, path):
        with open(path, 'rb') as f:
            obj = pkl.load(f)
        return obj

    def eval(self):

        g = []
        t = []
        for key, value in self.gold.items():
            t.append(value.predicate)
            g.append(self.prediction[key].predicate)
        #g = reduce(lambda x,y: x+y, [i['psuedo_label'] for i in self.data])
        #t = reduce(lambda x, y: x+y, [i['gold_predicates'] for i in self.data])
        return rand_score(g, t)

class PredicateData(Dataset):
    def __init__(self):
        self.data = {}

    def clear(self):
        self.data = {}
    def __getitem__(self, index):
        data = {
            'sentences': None
        }
        return super().__getitem__(index)

if __name__ == '__main__':
    dataset = Data(file_in='data/Newblocks/nb_train.pkl')
    print(dataset[0])
    print(len(dataset))
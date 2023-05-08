from copy import deepcopy
from functools import reduce

import numpy as np

from plan_generator.utils import check_key, locate_pred_in_state, match_diff_in_states, match_diff_in_states_, match_pred_in_state
from .Plan import Action, ActionModel, Predicate, State, Plan
from typing import Dict, List
import os


class Result():

    def __init__(self, **kwargs):
        self.__predicates = kwargs.get('predicates', [])
        self.__ams = kwargs.get('ams', {})
        self.plan_ids = kwargs.get('plan_ids', [])
        self.from_one_plan = kwargs.get('from_one_plan', True)
        self.__p_action_key = {}
        for action_name, am in self.__ams.items():
            self.__p_action_key[action_name] = {
                'add': am.add_list.content[:],
                'del': am.del_list.content[:],
                'precondition': am.precondition.content[:]
            }
    

    @property
    def p_action_key(self):
        return self.__p_action_key

    @property
    def predicates(self):
        return self.__predicates

    @property
    def ams(self):
        return self.__ams
        
    @classmethod
    def from_ams(cls, ams: dict):
        from_one_plan = False
        plan_ids = [-1]
        predicates = set()
        for am in ams.values():
            am: ActionModel
            predicates.update(am.del_list.content)
            predicates.update(am.add_list.content)
            predicates.update(am.precondition.content)
        predicates = list(predicates)
        info = {
            'predicates': predicates,
            'ams': ams,
            'plan_ids': plan_ids,
            'from_one_plan': from_one_plan
        }
        return cls(**info)

    @classmethod
    def parse(cls, res_file):
        _, file_name = os.path.split(res_file)
        plan_ids = file_name.split('|')[1: -1]
        plan_ids = [int(i) for i in plan_ids]
        from_one_plan = len(plan_ids) == 1
        info = {
            'predicates': cls.parse_Pred(res_file),
            'ams': cls.parse_AM(res_file),
            'plan_ids': plan_ids,
            'from_one_plan': from_one_plan

        }
        return cls(**info)


    @staticmethod
    def parse_Pred(res_file):
        predicates = []
        with open(res_file, 'r') as f:
            while True:
                stream = f.readline()
                if stream == '\n':
                    break
                elif stream in ['(define\n', '(:requirements :typing)\n', '(:types TYPE0 TYPE1)\n', '(:predicates\n', ')\n']:
                    continue
                elif stream.startswith('(:types'):
                    continue
                else:
                    args_type = {}
                    stream = stream.replace('(', '')
                    stream = stream.replace(')', '')
                    stream = stream.split(' ')
                    _stream = [stream[0]]
                    for idx, item in enumerate(stream[1:]):
                        if idx % 2 == 0:
                            arg = item
                            _stream.append(arg)
                        else:
                            type = item.replace('\n', '').replace('-', '').lower()
                            args_type[arg] = type
                    predicates.append(Predicate.parse(' '.join(_stream), args_type=args_type))

        return predicates

    @staticmethod
    def parse_AM(res_file) -> Dict:
        '''
        解析Result.out文件，得到action model
        :return: {am_name: am}，am_name为小写
        '''
        ams = {}
        '''
        parse action model
        '''
        with open(res_file, 'r') as f:
            action_stream = ''
            while True:
                stream = f.readline()
                if ':action' in stream:
                    action_stream += stream
                    break
            while True:
                stream = f.readline()
                if not stream:
                    break
                if ':action' in stream:
                    action_stream = ''
                    action_stream += stream
                elif stream == '\n':
                    am = ActionModel.parse(action_stream)
                    ams[am.name.lower()] = am
                    action_stream = ''
                else:
                    action_stream += stream
        return ams

    def output(self, path):
        def _output_action1(action: ActionModel):
            f.write(f'(:action {action.name.upper()}\n')
            f.write(f':parameters (')
            for arg, type in action.parameters.items():
                f.write(f'{arg} -{type.upper()} ')
            f.write(')\n:precondition(and')
            for p in action.precondition:
                f.write(f' ({p.name.upper()}')
                for param in p.params.param:
                    f.write(f' {param}')
                f.write(')')
            f.write(')\n')
            f.write(':effect (and')
            for p in action.del_list:
                f.write(' (not')
                f.write(f'({p.name}')
                for param in p.params.param:
                    f.write(f' {param}')
                f.write(')')
                f.write(')')
            for p in action.add_list:
                f.write(f' ({p.name}')
                for param in p.params.param:
                    f.write(f' {param}')
                f.write(')')
            f.write(')\n')
            f.write(')\n\n')
        
        
        def _output_predicate():
            for p in self.predicates:
                p: Predicate
                f.write(f'({p.name.upper()}')
                for param in p.params.param:
                    f.write(f' {param} -{p.params.type[param].upper()}')
                f.write(')\n')
            f.write(')')
        def _output_type():
            types = set()
            for item in self.predicates:
                types.update(list(item.params.type.values()))
            for item in types:
                item: str
                f.write(f' {item.upper()}')
            f.write(')\n')
        dir, file = os.path.split(path)
        if dir:
            os.makedirs(dir, exist_ok=True)
        with open(path, 'w') as f:
            f.write('(define\n(:requirements :typing)\n(:types')
            _output_type()
            f.write('(:predicates\n')
            _output_predicate()
            f.write(')\n\n')
            for action in self.__ams.values():
                _output_action1(action)
            
            f.write(')')



class Rectifier():

    def __init__(self, **kwargs):

        self.results = kwargs.get('results', [])[:]
        if self.results:
            self.results.sort(key=lambda x: x.plan_ids)
            self.actions = list(set(reduce(lambda x,y: x+y, [list(i.ams.keys()) for i in self.results])))

            '''self.__predicates = list(set(reduce(lambda x, y: x+y, [i.predicates for i in self.results])))
            for predicate in self.__predicates:
                predicate: Predicate
                if len(predicate.params.param) == 2 and predicate.params.type[predicate.params.param[0]] == predicate.params.type[predicate.params.param[1]]:
                    # 此处默认命题参数最多只有两个
                    self.__predicates.append(predicate.params_change([predicate.params[1], predicate.params[0]]))
          '''
        else:
            self.actions = []
            #self.__predicates = []

        self.plans = kwargs.get('plans', [])
        if self.plans:
            self.plans.sort(key=lambda x: x.plan_id)
        self.id2plan = {}
        for item in self.plans:
            self.id2plan[item.plan_id] = item

        
        '''self.__pred2id = {}
        for idx, item in enumerate(self.__predicates):
            self.__pred2id[item] = idx'''

        self.__p_action_key = {}
        self._p_action_key()
        self.__p2id_action_key = {}
        self._p2id_action_key()

    def _p2id_action_key(self):
        for action in self.actions:
            self.__p2id_action_key[action] = {'add': {}, 'del': {}, 'precondition': {}}
        for action_name, item in self.__p_action_key.items():
            for key in ['add', 'del', 'precondition']:
                for idx, predicate in enumerate(self.__p_action_key[action_name][key]):
                    self.__p2id_action_key[action_name][key][predicate] = idx


    def _p_action_key(self):
        for action_name in self.actions:
            self.__p_action_key[action_name] = {'add': set(), 'del': set(), 'precondition': set()}
        for result in self.results:
            result: Result
            for action_name in self.actions:
                for key in ['add', 'del', 'precondition']:
                    _result_action = result.p_action_key.get(action_name)
                    if not _result_action:
                        continue
                    self.__p_action_key[action_name][key].update(result.p_action_key[action_name][key])
                    
        for action_name in self.actions:
            for key in ['add', 'del', 'precondition']:
                temp = list(self.__p_action_key[action_name][key])
                self.__p_action_key[action_name][key] = temp



    @property
    def p2id_action_key(self):
        return self.__p2id_action_key


    @property
    def p_action_key(self):
        return self.__p_action_key
            
    '''@property
    def pred2id(self):
        Warning('self.pred2id is deprecated.')
        return deepcopy(self.__pred2id)

    @property
    def predicates(self):
        Warning('self.predicates is deprecated.')
        return self.__predicates'''

    def get_matrix(self, action, key):
        if action not in self.actions:
            raise ValueError(f'Unknown action {action}.')
        if key not in ['del', 'add', 'precondition']:
            raise NotImplementedError(f'Key should be either "precondition", "del" or "add", got {key}.')
        preds = []

        alpha = np.ones_like(self.results)
        for idx, result in enumerate(self.results):
            am = result.ams.get(action)
            if not am:
                preds.append([])
                alpha[idx] == 0
            else:
                if key == 'add':
                    l = am.add_list.content
                elif key == 'del':
                    l = am.del_list.content
                elif key == 'precondition':
                    l = am.precondition.content
                #l = [i.params_change(am.parameters) for i in l]
                preds.append(l)
        
        return self._get_matrix(preds, action, key), alpha

    def _get_matrix(self, preds, action, key):
            
        pred2id = self.p2id_action_key[action][key]
        ret = np.zeros((len(preds), len(pred2id)), dtype=int)
        for row, instance in enumerate(preds):
            if not instance:
                continue
            for p in instance:
                ret[row][pred2id[p]] = 1
        return ret

    def get_plan_by_id(self, id, return_index=False):
        return self.id2plan[id]


    def get_plan_by_id_(self, id, return_index=True):
        ret_id = -1
        for idx, plan in enumerate(self.plans):
            if plan.plan_id == int(id):
                ret_id = idx
                break
        if ret_id < 0:
            raise ValueError(f'plan id {id} not found.')
        if return_index:
            return ret_id
        return self.plans[ret_id]

    def locate(self, action, pred, key: str, plan: Plan):
        '''
        given a predicate schema, return state_id and pred_id
        '''
        if isinstance(pred, int):
            pred = self.predicates[pred]
        assert isinstance(pred, Predicate)
        
        placeholder2idx = {
            '?a': 0,
            '?b': 1,
            '?c': 2
        }


        if key not in ['add', 'del', 'precondition']:
            raise NotImplementedError(f'key shold be either "precondition", "add" or "del", got {key}.')
        
        state_id_list = []
        pred_instance_list = []
        pred_id_list = []
        _action_name_of_plan = [i.split(' ')[0][1:].lower() for i in plan.actions]
        _action_params_of_plan = [i.replace(')\n', '').split(' ')[1:] for i in plan.actions]
        action_name = action.name.lower()
        for idx, item in enumerate(_action_name_of_plan):
            if item == action_name:
                _action = action.params_change(_action_params_of_plan[idx])
                if key == 'del':
                    _temp = _action.del_list
                elif key == 'add':
                    _temp = _action.add_list
                elif key == 'precondition':
                    _temp = _action.precondition
                #pred_match, _ = match_pred_in_state(pred, _temp, _action.parameters)
                pred_match = pred.params_change([_action_params_of_plan[idx][placeholder2idx[i]] for i in pred.params.param])
                #if pred_match and _ >= 0:
                if pred_match:
                    _idx = idx + 1 if key == 'add' else idx
                    
                    pred_idx = locate_pred_in_state(pred_match, plan.states[_idx])
                    if pred_idx >= 0:
                        state_id_list.append(_idx)
                        pred_id_list.append(pred_idx)
                        pred_instance_list.append(pred_match)

                

        return state_id_list, pred_id_list, pred_instance_list
            
    def correct(self, action, pred, key: str, plan_id: int, gold_predicates: State):
        if isinstance(pred, int):
            pred = self.predicates[pred]
        assert isinstance(pred, list)

        plan = self.get_plan_by_id(plan_id, return_index=False)
        if key not in ['add', 'del', 'precondition']:
            raise NotImplementedError(f'key shold be either "precondition", "add" or "del", got {key}.')
        
        action_schema = action
        effect_list = action_schema.del_list if key == 'del' else action_schema.add_list
        if key == 'del':
            effect_list = action_schema.del_list
        elif key == 'add':
            effect_list = action_schema.add_list
        elif key == 'precondition':
            effect_list = action_schema.precondition
        pred_map = match_diff_in_states_(effect_list, gold_predicates, verbose=False)

        #_state_id_list, _pred_id_list, _src_list = self.locate(action, pred, key, plan)


        state_id_list = []
        pred_instance_list = []
        target_list = []
        src_list = []
        pred_id_list = []

        for item in pred:
            tgt = pred_map.get(item)
            if tgt is None:
                continue
            _state_id_list, _pred_id_list, _src_list = self.locate(action, item, key, plan)
            _sd_list, _pd_list, _s_list = self.locate(action, tgt, key, plan)
            for _tg_state_id in _sd_list:
                if _tg_state_id in _state_id_list:
                    _idx = _state_id_list.index(_tg_state_id)
                    _state_id_list.pop(_idx)
                    _pred_id_list.pop(_idx)
                    _src_list.pop(_idx)
            state_id_list += _state_id_list
            pred_id_list += _pred_id_list
            src_list += _src_list
            for _ in _src_list:
                target_list.append(tgt.params_change(_.params.param))
                
        

        assert len(state_id_list) == len(pred_id_list)
        assert len(state_id_list) == len(src_list)
        assert len(state_id_list) == len(target_list)   

        return state_id_list, pred_id_list, src_list, target_list
        

    def vec2predicates(self, vec, action, key):
        predicates = self.__p_action_key[action][key]
        ret = []
        for idx in range(len(vec)):
            if vec[idx] == 1:
                ret.append(predicates[idx])
        return ret


    def ratio2vec(self, ratio, ceil=0.5):
        assert ceil >= 0 and ceil <= 1
        return (ratio > ceil).astype(int)



    


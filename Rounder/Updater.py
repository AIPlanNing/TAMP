import copy
from functools import partial, reduce
import logging
import pickle as pkl

import numpy as np

from plan_generator.utils import error_rate, redundancy_rate
from torch import nn
import torch
from torch.optim.sgd import SGD
import os
from Rounder.utils import ActionModelModule
from dataset.dataset import PsuedoItem
from plan_generator.Plan import ActionModel, Plan, State
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    
)
logger = logging.getLogger(__name__)


from plan_generator.utils import check_key, generate_plan, match_wrong_predicate_in_gold, plan_match
from .Planner import Rectifier

from multiprocessing import Pool

class Updater():

    def __init__(self) -> None:
        pass


    def update(self):
        pass

class RecUpdater(Updater):
    
    def __init__(self, dataset, results, plans, device) -> None:
        super().__init__()
        self.device = device
        
        self.dataset = dataset
        self.__rec: Rectifier = Rectifier(results=results, plans=plans)

    @property
    def rec(self):
        return self.__rec

    @rec.setter
    def rec(self, results_plans):
        results, plans = results_plans
        self.__rec = Rectifier(results=results, plans=plans)

    

    def update(self, dataset, **kwargs):
        prediction_file = kwargs.get('prediction_file')
        ratio = kwargs.get('ratio', 0.7)
        if prediction_file is not None:
            pass
            #self.update_prediction(dataset, prediction_file)
        record, actions = self.get_record(ratio)
        record.sort(key=lambda x: x['id'])
        
        match_record = self.update_by_am(actions)
        print(len(record), len(match_record))
        dataset.dataset = {}
        self.commit_(dataset, record)
        self.commit_(dataset, match_record)
        
        return record, actions

    def update_prediction(self, dataset, prediction_file):
        with open(prediction_file, 'rb') as f:
            obj = pkl.load(f)
            for item in obj:
                plan_id, state_id, pred_id = item.id
                dataset.psuedo[(plan_id, state_id, pred_id)] = PsuedoItem(item.predicate, item.gold_params)


    def commit_(self, dataset, record):
        for item in record:
            plan_id, state_id, pred_id = item['id']
            # assert dataset.iloc(plan_id, state_id, pred_id) == (item['before'].name.lower(), [i.lower() for i in item['before'].params])
            #dataset.psuedo[(plan_id, state_id, pred_id)] = PsuedoItem(item['after'].name.lower(), [i.lower() for i in item['after'].params])
            dataset.dataset[(plan_id, state_id, pred_id)] = PsuedoItem(item['after'].name.lower(), [i.lower() for i in item['after'].params])
        return dataset

    def commit(self, dataset, record):
        dataset = copy.deepcopy(dataset)
        return self.commit_(dataset, record)

    
    def get_record(self, ratio=0.7):
        #multiprocess version
        record = []
        actions = {}
        ams = {}
        # train all g
        params = []
        action_name_key2g_m = self._get_g_m_full(self.rec.actions, ratio)# 8min
        for action_name in self.rec.actions:
            for key in ['add', 'del', 'precondition']:
                g = action_name_key2g_m[f'{action_name}_{key}']['g']
                m = action_name_key2g_m[f'{action_name}_{key}']['m']
                params.append([action_name, key, g, m])
        #ctx = torch.multiprocessing.get_context("spawn")
        '''pool = Pool(12)
        ret = pool.map(self._update_key_action, params)# 
        pool.close()
        pool.join()'''
        ret = [self._update_key_action(*i) for i in params]
        for idx, a_k_g_m in enumerate(params):
            action_name, key, _1, _2 = a_k_g_m
            record += ret[idx][0]
            actions[f'{action_name}_{key}'] = ret[idx][1]
        for action_name in self.rec.actions:
            precondition = actions[f'{action_name}_precondition']
            del_list = actions[f'{action_name}_del']
            add_list = actions[f'{action_name}_add']
            _idx = 0
            if action_name in ['putegginpan', 'putflourinpan', 'mix', 'removepanfromoven', 'bakecake', 'bakesouffle', 'cleanpan']:
                _idx = 5
            am = ActionModel.init_from(action_name, self.rec.results[_idx].ams[action_name.lower()].parameters, precondition, del_list, add_list)
            ams[action_name.lower()] = am
        

        logger.info(f'{actions}')
        # return record, actions
        return record, ams
    

    def _get_g_m_full(self, action_names: list, lr=1e-1, num_steps=25000, ratio=0.7):
        '''
        action_names: ['pick', 'move'...]
        return {
            action_name_key: {'m': m, 'g': g}
        }
        '''
        ret = {}
        inp = {}
        name_key2shape ={}
        for action_name in action_names:
            for key in ['add', 'del', 'precondition']:
                action_name_key = f'{action_name}_{key}'
                m, _ = self.rec.get_matrix(action_name, key)
                ret[action_name_key] = {'m': m, 'g': np.empty(0)}
                inp[action_name_key] = torch.from_numpy(m).to(self.device)
                name_key2shape[action_name_key] = m.shape
        model = ActionModelModule(name_key2shape=name_key2shape)
        model.to(self.device)
        optimizer = SGD(model.parameters(), lr)
        for epoch in range(num_steps):
            optimizer.zero_grad()
            loss = model(inp)
            if (epoch+1) %1000 == 0:
                logger.info(f'epoch: {epoch}, loss: {loss.item(): .6f}')
                pass
            loss.backward()
            optimizer.step()
        g = model.get_ratio()
        for name_key, _ratio in g.items():
            ret[name_key]['g'] = self.rec.ratio2vec(_ratio, ratio)[0]
        return ret
    


    def _update_key_action(self, *args):
        if len(args) == 1:
            action_name, key, g, m = args[0]
        else:
            action_name, key, g, m = args[0], args[1], args[2], args[3]
        pid = os.getpid()
        logger.info(f'{pid}: start')
        check_key(key)
        #g, m = self._get_g_m(key, action_name)
        gold_predicates = self.rec.vec2predicates(g, action_name, key)
        s_gold_predicates = State.predicate_init(gold_predicates)
        record = []
        actions = {}
        actions[f'{action_name}_{key}'] = gold_predicates

        
        
        for idx, item in enumerate(g - m):
            plan_id = self.rec.results[idx].plan_ids[0]
            preds2b_modify = self.rec.vec2predicates(-item, action_name, key)
            
            action = self.rec.results[idx].ams.get(action_name)
            if preds2b_modify:
                ret = self.rec.correct(action, preds2b_modify, key, plan_id, s_gold_predicates)
                # 一条plan中可能有多个action所以返回一个列表
                for s_id, p_id, src, tgt in zip(ret[0], ret[1], ret[2], ret[3]):
                    if p_id >= 0:
                        

                        record.append({
                            'before': src,
                            'after': tgt,
                            'id': (plan_id, s_id, p_id)
                        })
        logger.info(f'{pid}: done')
        return record, gold_predicates
    

    def _get_g_m(self, key, action_name, lr=1e-1, num_steps=25000, ratio=0.7):
        m, _alpha = self.rec.get_matrix(action_name, key)
        model = nn.ModuleDict(
            {
                'alpha': nn.Linear(1, m.shape[0]),
                'beta': nn.Linear(m.shape[1], 1)
            }
        )

        model.to(self.device)

        optimizer = SGD(model.parameters(), lr)
        mm = torch.from_numpy(m).to(self.device)
        for epoch in range(num_steps):
            optimizer.zero_grad()
            loss1 = torch.norm(model['alpha'].weight.sigmoid().mm(model['beta'].weight.sigmoid())-mm)
            loss2 = (model['beta'].weight.sigmoid().sum() * model['alpha'].weight.sigmoid() - mm.sum(1).unsqueeze(1)).pow(2).mean()
            loss = loss1 + loss2
            if (epoch + 1) % 1000 == 0:
                #logger.info(f'epoch: {epoch}, loss: {loss.item(): .6f}')
                pass
            loss.backward()
            optimizer.step()
        
        g = model['beta'].weight.sigmoid().cpu().detach().numpy()
        g = self.rec.ratio2vec(g, ratio)[0]
        return g, m
    

    def update_by_plan_match(self, gold_plan: Plan, raw_plan: Plan, plan_id: int):
        pred_idx, preds = plan_match(gold_plan, raw_plan)
        record = []
        for s_id, p_id_list, pred_list in zip(range(len(pred_idx)), pred_idx, preds):
            for p_id, pred in zip(p_id_list, pred_list):
                record.append(
                    {
                        'before': None,
                        'after': pred,
                        'id': (plan_id, s_id+1, p_id)
                    }
                )
        return record
            

        
    def update_by_am(self, ams):
        
        record = []
        def _precess(ams, plan):
            gold_plan = generate_plan(ams, plan.actions, plan.states[0], plan.args_type)
            ret = self.update_by_plan_match(gold_plan, plan, plan.plan_id)
            return ret
        fn = partial(self.__process, ams=ams)
        '''pool = Pool(16)
        record = pool.map(fn, self.rec.plans,)
        pool.close()
        pool.join()'''
        record = [fn(plan=i) for i in self.rec.plans]
        record = reduce(lambda x, y: x+y, record)
        return record
    

    def __process(self, ams, plan):
        pid = os.getpid()
        #logger.info(f'{pid}: __process start.')
        gold_plan = generate_plan(ams, plan.actions, plan.states[0], plan.args_type)
        
        pred_idx, preds = plan_match(gold_plan, plan)
        record = []
        plan_id = plan.plan_id
        for s_id, p_id_list, pred_list in zip(range(len(pred_idx)), pred_idx, preds):
            for p_id, pred in zip(p_id_list, pred_list):
                record.append(
                    {
                        'before': None,
                        'after': pred,
                        'id': (plan_id, s_id+1, p_id)
                    }
                )
        #logger.info(f'{pid}: __process done.')
        return record
    


        

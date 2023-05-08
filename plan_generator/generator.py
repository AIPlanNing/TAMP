# from .Action import ActionModel, Action
import os, re, copy
from model.metric import AMMetric
from plan_generator.Plan import Plan, State, Predicate, Action, ActionModel
from .Result import Result, Rectifier
from typing import Dict, List
import numpy as np
from model.dataset.data_reader import StateData
import glob
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
logger = logging.getLogger(__name__)
from functools import reduce
from config.nb_config import args_type
from .utils import match_pred_in_state, locate_pred_in_state


def predicate_type_equal(instance: Predicate, schema: Predicate, args_type):
    if not instance.name.lower() == schema.name.lower():
        return False
    if [i.lower() for i in schema.params] == [args_type[i].lower() for i in instance.params]:
        return True
    return False
    
    


def check_AC(am):
            '''
            check action constraint
            '''
            add_and_pre = am.add_list & am.precondition
            if not add_and_pre.is_empty():
                print(f'{am.name} action constraint: add list in precondition {add_and_pre}')
            del_sub_pre = am.del_list - am.precondition
            if not del_sub_pre.is_empty():
                print(f'action constraint: del list not in precondition {del_sub_pre}')
            if add_and_pre.is_empty() and del_sub_pre.is_empty():
                print(f'{am.name} action constraint fulfill.')

def check_PC(am):
    '''
    check plan constraint
    '''
    pass






def parse_AM(res_file, regular=False) -> Dict:
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
                if regular:
                    am = am.regular()
                ams[am.name.lower()] = am
                action_stream = ''
            else:
                action_stream += stream
    return ams

         
def parse_result(res_file, **kwargs):
    '''
    解析HTNML生成的Result.out文件得到action model，将action model作用于Soln文件生成
    新的plan
    :param res_file: result.out文件
    :param soln_dir: 存放soln文件的路径，该路径下的soln文件名形如 Soln0，数字后缀由0
    递增到soln_num-1
    :param soln_num: soln文件的个数
    :return plans(list(Plan)): 生成的plan list
    '''
    soln_dir = kwargs.get('soln_dir')
    soln_files = kwargs.get('soln_files')
    soln_num = kwargs.get('soln_num')
    if not soln_dir and not soln_files:
        raise ValueError('soln_dir or soln_files should be provided.')
    if soln_dir and soln_files:
        raise Warning('soln_dir and soln_files are both provided, use soln_files.')
    
    soln_files = soln_files if soln_files else glob.glob(os.path.join(soln_dir, 'Soln*'))
    soln_files = soln_files[:soln_num] if soln_num else soln_files
    ams = parse_AM(res_file)
    plans = []
    for file in soln_files:
        plan = Plan.parse(file)
        '''
        plan的第一个动作根据action model需要满足一个前置条件，该前置条件有可能与
        plan的初始状态相异：
            若前置条件存在初始状态所没有的命题，那么在新生成的plan中，在plan的初
            始状态增加该命题；
        然后根据该初始状态和action model，获得整条新的plan
        '''
        # metric = AMMetric(list(ams.values()))
        init_flag = 1
        for idx, action_stream in enumerate(plan.actions):
            action_stream = action_stream.replace('\n', '')
            action_stream = action_stream.replace(')', '')
            action_stream = action_stream.replace('(', '')
            action_name = action_stream.split(' ')[0].lower()
            am = ams[action_name]
            params = action_stream.split(' ')[1:]
            params_map = {}
            for k, p in zip(am.parameters.keys(), params):
                params_map[k] = p
            am = Action.instantiate_from(ams[action_name], params_map)
            am = am.regular()
            '''
            am的precondition与plan.states[0]并集
            '''
            if init_flag:
                # 创建init state
                plan[0] = plan[0] | am.precondition
                init_flag = 0
            '''
            更新
            由于动作模型是不正确的，所以由初始状态和动作生成整条plan的时候，
            action与其前面的状态会产生矛盾，因而action需要要求状态进行增删，
            显而易见，最后增删的内容都要作用到初始状态上
            但是可能出现第i+1个动作要求删除命题p，但是p又存在于第i个动作的
            add_list中，那么如何处理呢？TODO
            '''
            '''
            由plan[idx]与action得到next_state、to_be_add和to_be_del
            '''
            ret = plan[idx](am)
            next_state = ret['next_state']
            to_be_del = ret['to_be_del']
            to_be_add = ret['to_be_add']
            plan[0] = plan[0] | to_be_add
            plan[0] = plan[0] - to_be_del
            plan[idx+1] = next_state
            assert '' not in reduce(lambda x, y: x+y, [[i.name for i in j.content] for j in plan.states])
        plans.append(plan)
    

    return plans


def res2dataset(soln_dir='tmp/nb/solutions/solution', res_dir='HTNML/am1', dataset_prototype='data/Newblocks/nb_train.pkl', specific_res_file=None) -> StateData:
    '''
    result file命名方式为"Result.out|{id1}|{id2}|{id3}|...|"
    solution file命名方式为"Soln{id}"
    '''
    dataset = StateData(dataset_prototype)
    res_file_list = glob.glob(os.path.join(res_dir, 'Result.out|*'))
    plans = []
    for res_file in res_file_list:
        
        soln_ids = res_file.split('|')[1:-1]
        soln_files = ['Soln' + i for i in soln_ids]
        soln_files = [os.path.join(soln_dir, i) for i in soln_files]
        res_file = specific_res_file if specific_res_file else res_file
        plans += parse_result(res_file, soln_files=soln_files)
    plans.sort(key=lambda x: x.plan_id)
    dataset.data.sort(key=lambda x: (x['plan_idx'], x['trace_idx']))
    # states = reduce(lambda x, y: x+y, [i.states for i in plans])
    data_idx = 0
    for plan in plans:
        plan_id = plan.plan_id
        plan_id = int(plan_id)
        for state_id, state in enumerate(plan.states):
            # state_in_dataset = dataset.data[data_idx]
            
            assert dataset.data[data_idx]['trace_idx'] == state_id
            assert dataset.data[data_idx]['plan_idx'] == plan_id
            dataset.data[data_idx]['predicates'] = [i.name.lower() for i in state.content]
            assert '' not in dataset.data[data_idx]['predicates']
            dataset.data[data_idx]['gold_params_list'] = [i.params for i in state.content]
            data_idx += 1
    # dataset.labels = None
    
    return dataset
            
    
def eval_action_model(res_dir, soln_dir, verbose=False):
    logger.info('evaluating result files...')
    res_file_list = glob.glob(os.path.join(res_dir, 'Result.out|*'))
    error_rate_list = []
    action_constraint_list = []
    best_res, best_error_rate, best_action_constraint = None, 1.0, 1.0
    for res_file in res_file_list:
        
        soln_files = glob.glob(os.path.join(soln_dir, 'Soln*'))
        soln_files = soln_files[:min(len(soln_files), 100)]

        total_error_rate = 0.0
        ams = parse_AM(res_file, regular=False)
        metric = AMMetric(list(ams.values()))
        action_constraint_score = metric.eval_action_constraint(gather='mean')
        action_constraint_list.append(action_constraint_score)
        for soln_file in soln_files:
            plan = Plan.parse(soln_file)
            error_rate = metric.eval_error_rate(plan, gather='mean')
            total_error_rate += error_rate
        total_error_rate /= len(soln_files)
        error_rate_list.append(total_error_rate)
        if verbose:
            print(f'{res_file}\n\tac_score: {action_constraint_score: .2f}\terror_rate: {total_error_rate: .2f}, len: {len(soln_files)}')
        if total_error_rate < best_error_rate:
            best_error_rate = total_error_rate
            best_res = res_file
    if verbose:
        print(f'--------\t\t result \t\t---------')
        print(f'evaluate {len(res_file_list)} results, avg action constraint disobey: {sum(action_constraint_list)/len(action_constraint_list): .6f}, avg error rate: {sum(error_rate_list)/ len(error_rate_list): .6f}')
        
        print(f'best res file: {best_res}, best error rate: {best_error_rate: .6f}')
        print(f'--------\t\t--------\t\t---------')
    return best_res, best_error_rate








def get_matrix(preds, map: dict):

    ret = np.zeros((len(preds), len(map)), dtype=int)
    for row, instance in enumerate(preds):
        if not instance:
            continue
        for p in instance:
            ret[row][map[p]] = 1
    return ret



if __name__ == '__main__':
    dataset = res2dataset()
    pass

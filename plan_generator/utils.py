from .Plan import Plan, State, Predicate, ActionModel, Action
import copy
import logging


def locate_pred_in_state(pred, state):
    '''
    pred be like (ON block1 block2)
    '''
    idx_list = []
    for idx, item in enumerate(state):
        if item == pred:
            idx_list.append(idx)
    # assert len(idx_list) < 2
    if len(idx_list) > 1:
        return -2
    if len(idx_list) == 0:
        return -1
    return idx_list[0]


def match_pred_in_state(pred, state, args_type):
    '''
    pred be like (ON type0, type0) # 0103 +++pred (ON ?a ?b)
    没有考虑如下情况: (ON a, b) (ON b, a)同时出现在一个状态的情况# 0103 +++考虑了
    返回第一个匹配的情况
    '''
    flag = False
    for idx, item in enumerate(state):
        if pred.name.lower() == item.name.lower():
            if len(item.params) == len(pred.params):
                for i, j in zip(item.params, pred.params):
                    if not args_type[i].lower() == j.lower():
                        flag = False
                        break
                    flag = True
                # pred.params = item.params[:]
                if flag:
                    return copy.deepcopy(item), idx
    return None, -1


def match_diff_in_states(inp: State, gold: State, verbose=True):
    '''
    inp and gold be like: State(NOTHING['type0'], ...)
    ++ 0103 inp State(NOTHING['?a'], ...)
    '''
    i_m_g = inp - gold # wrong predicates
    g_m_i = gold - inp
    match = {}
    for key in i_m_g:
        # 找到与key唯一对应的value
        # 如果g_m_i为空，那么说明该动作模型是由多个动作学到的，则在整个
        temp = []
        
        for value in gold:
            if len(key.params) == len(value.params) and set(key.params) == set(value.params):
                temp.append(value)
        if not len(temp) == 1:
            temp = []
            for value in g_m_i:
                if len(key.params) == len(value.params) and set(key.params) == set(value.params):
                    temp.append(value)
        if len(temp) == 1:
            match[key] = temp[0]
        elif len(temp) == 0:
            if verbose:
                print(f'{key} match 0.(effect比target多了一项，说明该动作模型由多个动作学得)')
        elif len(temp) > 1:
            if verbose:
                print(f'{key} match {temp}.')
    return match
            

def match_diff_in_states_(inp: State, gold: State, verbose=True):
    # 还在修改。。。。。。
    
    '''
    inp and gold be like: State(NOTHING['type0'], ...)
    ++ 0103 inp State(NOTHING['?a'], ...)
    '''
    i_m_g = inp - gold # wrong predicates
    g_m_i = gold - inp
    match = {}
    for key in i_m_g:
        # 找到与key唯一对应的value
        # 如果g_m_i为空，那么说明该动作模型是由多个动作学到的，则在整个
        temp = []
        key: Predicate
        
        for value in gold:
            value: Predicate
            if key.params.name_type_equal(value.params):
                temp.append(value)
        if not len(temp) == 1:
            temp = []
            for value in g_m_i:
                if key.params.name_type_equal(value.params):
                    temp.append(value)
        if len(temp) == 1:
            match[key] = temp[0]
        elif len(temp) == 0:
            if verbose:
                print(f'{key} match 0.(effect比target多了一项，说明该动作模型由多个动作学得)')
        elif len(temp) > 1:
            if verbose:
                print(f'{key} match {temp}.')
    return match


def match_wrong_predicate_in_gold(wrong_pred, gold_preds):
    temp = []

    for idx, gold_pred in enumerate(gold_preds):
        if gold_pred.params == wrong_pred.params:
            temp.append(gold_pred)
            assert len(gold_pred.params) == len(wrong_pred.params)
            for i, j in zip(gold_pred.params, wrong_pred.params):
                assert i == j
    return temp


def check_key(key):
    if key not in ['add', 'del', 'precondition']:
        raise NotImplementedError(f'key shold be either "precondition", "add" or "del", got {key}.')
    return key  
    
                    
def generate_plan(models: dict, trace: list, init: State, args_type):
    '''
    item of trace: (move loc-1-1 loc-2-3)\n
    '''
    content = [init]
    next_state = init
    for action_name in trace:
        action = remove_special(action_name).split(' ')
        action_name, params = action[0], action[1:]
        am: ActionModel = models[action_name.lower()].params_change(params)
        next_state = next_state - am.del_list | am.add_list
        content.append(next_state)
    return Plan.init_from_action_states(trace, content, args_type)
    

def error_rate(ams: dict, plan: Plan):
    assert isinstance(ams, dict)
    # assert isinstance(ams.values()[0], ActionModel)
    assert isinstance(plan, Plan)
    prec_len = 0
    prec_disobey_len = 0
    for state, action_name in zip(plan.states, plan.actions):
        action = remove_special(action_name).split(' ')
        action_name, params = action[0], action[1: ]
        precondition = ams[action_name].params_change(params).precondition
        prec_len += len(precondition)
        prec_disobey_len += len(precondition-state)
    rate = prec_disobey_len / (prec_len+1e-4)
    assert rate >= 0
    return rate

def redundancy_rate(ams: dict, plan: Plan):
    assert isinstance(ams, dict)
    #assert isinstance(ams[0], ActionModel)
    assert isinstance(plan, Plan)
    add_len = 0
    not_in_prec_len = 0
    actions = plan.actions[:]
    actions.reverse()
    state = State.predicate_init([])
    for idx, action_name in enumerate(actions):
        action = remove_special(action_name).split(' ')
        action_name, params = action[0], action[1:]
        am = ams[action_name].params_change(params)
        add_list = am.add_list
        if idx > 0:
            add_len += len(add_list)
            not_in_prec_len += len(add_list-state)
        
        state.content += am.precondition

    rate = not_in_prec_len / (add_len+1e-4)
    return rate



    
def plan_match(gold_plan: Plan, plan: Plan):
    '''
    给定两个plan，找出能对应上的命题
    对应规则：1. 在gold_plan和plan对应的状态中有相同的命题
             2. 该命题在plan的这个状态中唯一存在
    形式化表达：
    s_i \in plan, s'_i \in gold_plan, i \in {0, 1, ..., len(plan)},
    \exist p_j \in s_i && p_j \in s'_i
    &&
    p_j \not \in s_i - p_j
    ret: pred_id(list) (of plan, not gold_plan)
    '''
    assert len(gold_plan) == len(plan)
    state_idx = []
    pred_idx = []
    preds = []
    

    for s_id, gold_state, state in zip(range(len(plan)-1), gold_plan.states[1:], plan.states[1:]):
        pred_idx.append([])
        preds.append([])
        _g_pred_idx = []
        for p_id, pred in enumerate(state):
            for g_p_id, gold_pred in enumerate(gold_state):
                if pred == gold_pred and g_p_id not in _g_pred_idx:
                    '''
                    plan中的某一状态s的某一条命题与gold中的某一条命题相同，
                    且在当前状态s中，该命题是唯一的，则添加
                    '''
                    pred_idx[s_id].append(p_id)
                    preds[s_id].append(pred)
                    _g_pred_idx.append(g_p_id)
        
    return pred_idx, preds


    
    
def remove_special(stream, special=['(', ')', '\n']):
    ret = stream
    for token in special:
        ret = ret.replace(token, '')
    return ret
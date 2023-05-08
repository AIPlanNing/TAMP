
from plan_generator.Plan import Plan, State, Predicate, Goal
import os
import pickle as pkl
import argparse


class SolutionGenerator():
    def __init__(self, args_type, file_out_prefix='Soln') -> None:
        self.file_out_prefix = file_out_prefix
        self.args_type = args_type
        
    

    def generate(self, dir_out, use_gold_params=False, **kwargs):
        raw_data = kwargs.get('raw_data')
        if not raw_data:
            raw_data_file = kwargs.get('raw_data_file')
            if not raw_data_file:
                raise ValueError(f'raw_data or raw_data_file should be provided.')
            raw_f = open(raw_data_file, 'rb')
            raw_data = pkl.load(raw_f)
            raw_f.close()

        predictions = kwargs.get('predictions')
        if not predictions:
            prediction_file = kwargs.get('prediction_file')
            if not prediction_file:
                raise ValueError(f'predictions or prediction_file should be provided.')
            pred_f = open(prediction_file, 'rb')
            predictions = pkl.load(pred_f)
            pred_f.close()
        raw_data.sort(key=lambda x: x['id'])
        ret = self._generate(raw_data, predictions, dir_out, use_gold_params)
        return ret
        

    def _generate(self, raw_data, predictions, dir_out, use_gold_params=False):
        
        global_idx = 0
        plans = []
        for data in raw_data:
            states = []
            plan_id = data['id']
            actions = [None] + data['plan'][:-1] + [None]
            info = {}
            info['plan_id'] = data['id']
            info['actions'] = data['plan'][:-1]
            state_num = len(data['text_trace'])
            init = None
            gold_goal = data['goal_state']
            goal = []
            for state_idx in range(state_num):
                state_info = {}
                predicates = []
                pred_num = len(data['text_trace'][state_idx].split('.')) - 1
                for pred_idx in range(pred_num):
                    one_prediction = predictions[global_idx]
                    global_idx += 1
                    predicate = Predicate(one_prediction, use_gold_params=use_gold_params)
                    assert one_prediction.id[0] == info['plan_id']
                    assert one_prediction.id[1] == state_idx
                    assert one_prediction.id[2] == pred_idx
                    if one_prediction.is_goal:
                        goal.append(predicate)
                    predicates.append(predicate)
                state_info['predicates'] = predicates
                state_info['plan_id'] = info['plan_id']
                state_info['state_id'] = state_idx
                state_info['pre_action'] = actions[state_idx]
                state_info['post_action'] = actions[state_idx+1]
                state = State(state_info)
                states.append(state)
            goal_info = {
                'predicates': goal, 
                'plan_id': plan_id
            }
            
            info['state_list'] = states
            info['goal'] = Goal(goal_info)
            plan = Plan(info, self.args_type)
            plans.append(plan)
            plan.output(os.path.join(dir_out, f'{self.file_out_prefix}{plan_id}'))



        return plans
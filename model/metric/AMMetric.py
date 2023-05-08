
from plan_generator.Plan import Plan, State, Action, ActionModel, Predicate


class AMMetric:

    def __init__(self, ams: list) -> None:
        self.action_models = ams
        self.gather_fn_dict = {
            'mean': lambda x: sum(x)/len(x), 
            'sum': sum
        }

    def eval_error_rate(self, plan: Plan, verbose=False, gather='mean'):
        '''
        
        '''
        error = {}
        error_rate = {}
        
        # 检查plan的action是否在self.action_models
        for item in plan.actions:
            item = item.replace('(', '')
            item = item.replace(')', '')
            item = item.split(' ')[0]
            if not item.lower() in [i.name.lower() for i in self.action_models]:
                raise ValueError(f'action {item} in plan but not in given action models.')
        # plan中除初始状态的每一个状态中的
        for idx, action in enumerate(plan.actions):
            action = action.replace('(', '')
            action = action.replace(')', '')
            action = action.split(' ')[0].lower()
            _idx = [i.name.lower() for i in self.action_models].index(action)
            am: ActionModel = self.action_models[_idx]
            if idx == 0:
                prcd = plan[idx]
                # error_state = am.precondition - plan[idx] # predicate in am's precondition while not in pre state
            else:
                prcd = plan[0] | self.action_models[0].precondition
                for __idx in range(0, idx):
                    action = plan.actions[__idx]
                    action = action.replace('(', '')
                    action = action.replace(')', '')
                    action = action.split(' ')[0].lower()
                    _idx = [i.name.lower() for i in self.action_models].index(action)
                    prcd = prcd - self.action_models[_idx].del_list
                    prcd = prcd | self.action_models[_idx].add_list
            error_state = am.precondition - prcd
            rate = len(error_state) / len(am.precondition)
            assert rate >= 0.0 and rate <= 1.0
            error[idx] = error_state
            error_rate[idx] = rate
        if verbose:
            print(f'--------\t eval plan: {plan.actions} \t--------')
            for idx, action in enumerate(plan.actions):
                action = action.replace('(', '')
                action = action.replace(')', '')
                action = action.split(' ')[0].lower()
                _idx = [i.name.lower() for i in self.action_models].index(action)
                print(f'#{idx} action in plan: {action}, error rate: {error_rate[idx]}')
                print(f'\tprecondition of the action: {self.action_models[_idx].precondition}')
                print(f'\tpredicates appear in preconditon while not in prestate or generate by previous action: {error[idx]}')
                print(f'\tnumber of predicate in precondition: {len(self.action_models[_idx].precondition)}')
                print(f'\tnumber of error in precondtion: {len(error[idx])}')
            print(f'--------\t eval done. \t--------')
        gather_fn = self.gather_fn_dict[gather]
        ret = gather_fn(list(error_rate.values()))
        return ret

    def eval_action_constraint(self, weight_add=1, weight_del=1, weight_pre=1, verbose=False, gather='mean'):
        contradictory = []
        contra_rate = []
        for idx, am in enumerate(self.action_models):
            del_add_contra = am.del_list & am.add_list
            del_pred_contra = am.del_list - am.precondition
            add_pred_contra = am.add_list & am.precondition
            contradictory.append({
                'del_add': del_add_contra,
                'del_pred': del_pred_contra,
                'add_pred': add_pred_contra
            })
            contra_rate.append({
                'del_add': len(del_add_contra) / len(am.del_list | am.add_list),
                'del_pred': len(del_pred_contra) / len(am.del_list),
                'add_pred': len(add_pred_contra) / len(am.add_list)
            })
            
        if verbose:
            print(f'--------\t eval action constraint \t--------')
            for idx, am in enumerate(self.action_models):
                print(f'eval {am.name}')
                print(f'\t length of del list: {len(am.del_list)}')
                print(f'\t length of add list: {len(am.add_list)}')
                print(f'\t length of precondition: {len(am.precondition)}')
                print(f'\t number of del_add error: {len(contradictory[idx]["del_add"])}')
                print(f'\t number of del_pred error: {len(contradictory[idx]["del_pred"])}')
                print(f'\t number of add_pred error: {len(contradictory[idx]["add_pred"])}')
                print(f'\t contradictory rate: del_add {contra_rate[idx]["del_add"]}, del_pred {contra_rate[idx]["del_pred"]}, add_pred: {contra_rate[idx]["add_pred"]}')
            print(f'--------\t eval done. \t--------')
        _sum = sum([weight_del, weight_add, weight_pre])
        weight_pre, weight_add, weight_del = weight_pre/_sum, weight_add/_sum, weight_del/_sum
        gather_fn = self.gather_fn_dict[gather]
        score = [weight_pre*i['del_add'] + weight_del*i['del_pred'] + weight_add*i['add_pred'] for i in contra_rate]
        return gather_fn(score)

            
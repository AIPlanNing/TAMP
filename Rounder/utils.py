
from copy import deepcopy
import torch
import torch.nn as nn

from plan_generator.utils import error_rate, redundancy_rate


def _make_map(keys, values):
    ret = {}
    if isinstance(values, dict):
        for item in keys:
            ret[item] = values[item]

    elif isinstance(values, list):
        assert len(values) == len(keys)
        for key, value in zip(keys, values):
            ret[key] = value
    
    else:
        raise ValueError(f'init should be list or dict')
    return ret


class LossScheduler():

    def __init__(self, loss_names: list, init, delta=0, floor=0, ceil=100) -> None:
        self.__loss = loss_names
        self.__loss_weight = _make_map(loss_names, init)
        
        if isinstance(delta, int):
            self.__loss_delta = _make_map(loss_names, [delta]*len(loss_names))
        else:
            self.__loss_delta = _make_map(loss_names, delta)
        
        if isinstance(floor, int):
            self.__loss_floor = _make_map(loss_names, [floor]*len(loss_names))
        else:
            self.__loss_floor = _make_map(loss_names, floor)

        if isinstance(ceil, int):
            self.__loss_ceil = _make_map(loss_names, [ceil]*len(loss_names))
        else:
            self.__loss_ceil = _make_map(loss_names, ceil)



    def step(self):
        for key, value in self.__loss_weight.items():
            if self.__loss_delta[key] == 0:
                continue
            elif self.__loss_delta[key] > 0:
                self.__loss_weight[key] = min(self.__loss_ceil[key], value+self.__loss_delta[key])
            else:
                self.__loss_weight[key] = max(self.__loss_floor[key], value+self.__loss_delta[key])

        
    @property
    def loss_weight(self):
        return deepcopy(self.__loss_weight)

    @property
    def loss_delta(self):
        return deepcopy(self.__loss_delta)

    @property
    def loss_floor(self):
        return deepcopy(self.__loss_floor)

    @property
    def loss_ceil(self):
        return deepcopy(self.__loss_ceil)



class ActionModelModule(nn.Module):
    def __init__(self, name_key2shape:dict):
        super().__init__()
        self.action_name_key2module = nn.ModuleDict()
        for action_name_key, shape in name_key2shape.items():
            if shape[1] == 0:
                continue
            self.action_name_key2module[action_name_key] = nn.ModuleDict(
                {
                    'alpha': nn.Linear(1, shape[0]),
                    'beta': nn.Linear(shape[1], 1)
                }
            )
    
    def forward(self, inp):
        # inp: {action_name_key: matrix}
        losses = 0.0
        for action_name_key, module in self.action_name_key2module.items():
            mm = inp[action_name_key]
            mul = module['alpha'].weight.sigmoid().mm(module['beta'].weight.sigmoid()) - mm
            loss = torch.norm(mul)
            losses += loss
        return losses

    def get_ratio(self):
        ret = {}
        for action_name_key, module in self.action_name_key2module.items():
            ret[action_name_key] = module['beta'].weight.sigmoid().cpu().detach().numpy()
        return ret
        

def eval_action_model(ams: dict, plans: list):
    lst_error_rate = [error_rate(ams, p) for p in plans]
    lst_redundancy_rate = [redundancy_rate(ams, p) for p in plans]

    return sum(lst_error_rate)/len(lst_error_rate), sum(lst_redundancy_rate)/len(lst_redundancy_rate)
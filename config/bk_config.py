args_type = {
    'soap-0': 'type0',
    'soap-1': 'type0',
    'soap-2': 'type0',
    'soap-3': 'type0',

    'new-0': 'type1',
    'new-1': 'type1',
    'new-2': 'type1',
    'new-3': 'type1',

    'oven-0': 'type2',
    'oven-1': 'type2',

    'pan-0': 'type3',
    'pan-1': 'type3',
    'pan-2': 'type3',
    'pan-3': 'type3',

    'egg-0': 'type4',
    'egg-1': 'type4',
    'egg-2': 'type4',

    'flour-0': 'type5'
}


def hash_param(param, id2token=None):
    if isinstance(param, int):
        if param == 0:
            return 0
        token = id2token[param]
        token_t = args_type[token]
        return {'type0': 0, 'type1': 1, 'type2': 2, 'type3': 3, 'type4': 4, 'type5': 5}[token_t]
    elif isinstance(param, str):
        token = param
        token_t = args_type[token]
        return {'type0': 0, 'type1': 1, 'type2': 2, 'type3': 3, 'type4': 4, 'type5': 5}[token_t]
    elif isinstance(param, list):
        return [hash_param(i, id2token) for i in param]
    else:
        raise ValueError()

def get_param_type(param):
    

    if isinstance(param, int):
        return param

    if isinstance(param, list):
        if isinstance(param[0], int):
            return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[sum(param)]
        elif isinstance(param[0], list):
            return [get_param_type(i) for i in param]

    else:
        raise ValueError()
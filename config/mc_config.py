model_config_adjust_list = ['vocab_size']

model_config = {
    'embd_hid': 128, 
    'hidden': 128,
    'dropout': 0.1,
    'encoder_num': 2,
    'encoder_name': 'gru',
    'embedding_from_wv': True,
    'wv_model_file': 'tmp/mc/wv/mc_wv.pkl',
    'embedding_freeze': True, 
    'state_encoder_type': 'rnn', # rnn | avg
    'temperature': 1,
}


training_config_adjust_list = ['device']
training_config = {
    'pad_id': 0,
    'batch_size': 128, 
    'lr': 1e-3,
    'weight_decay': 0.0001,
    'epoch': 20, 
    'log_steps': 5, 
    'prediction_out_file': 'tmp/mc/predictions/prediction.pkl',
    'data_file': 'data/Minecraft/mc_dev.pkl'

}


args_type = {
    'log-0': 'type0',
    'log-1': 'type0',
    'log-2': 'type0',
    'grass-3': 'type0',
    'log-4': 'type0',
    'grass-5': 'type0',
    'new-0': 'type0',
    'new-1': 'type0',
    'new-2': 'type0',
    'agent': 'type1',
    'loc-0-0': 'type2',
    'loc-0-1': 'type2',
    'loc-0-2': 'type2',
    'loc-0-3': 'type2',
    'loc-1-0': 'type2',
    'loc-1-1': 'type2',
    'loc-1-2': 'type2',
    'loc-1-3': 'type2',
    'loc-2-0': 'type2',
    'loc-2-1': 'type2',
    'loc-2-2': 'type2',
    'loc-2-3': 'type2',
}

label2word = {
    0: 88, # euquipped
    1: 2, # recall
    2: 71, # at
    3: 81, # storehouse
    4: 56, # nothing
    5: 48, # free
    6: 53, # agent
    7: 34, # move
    8: 73, # pick
    9: 61, # making
    10: 89, # plank
    11: 44, # grass
    12: 57, # equip
    13: 42, # log
}


def hash_param(param, id2token=None):
    if isinstance(param, int):
        if param == 0:
            return 0
        token = id2token[param]
        token_t = args_type[token]
        return {'type0': 1, 'type1': 3, 'type2': 5}[token_t]
    elif isinstance(param, str):
        token = param
        token_t = args_type[token]
        return {'type0': 1, 'type1': 3, 'type2': 5}[token_t]
    elif isinstance(param, list):
        return [hash_param(i, id2token) for i in param]
    else:
        raise ValueError()

def get_param_type(param):
    

    if isinstance(param, int):
        return param

    if isinstance(param, list):
        if isinstance(param[0], int):
            return {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}[sum(param)]
        elif isinstance(param[0], list):
            return [get_param_type(i) for i in param]

    else:
        raise ValueError()
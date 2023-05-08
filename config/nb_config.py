model_config_adjust_list = ['vocab_size']

model_config = {
    'embd_hid': 128, 
    'hidden': 128,
    'dropout': 0.1,
    'encoder_num': 2,
    'encoder_name': 'gru',
    'embedding_from_wv': True,
    'wv_model_file': 'tmp/nb/wv/wv.model',
    'embedding_freeze': True, 
    'state_encoder_type': 'rnn', # rnn | avg
    'temperature': 0.1,
}


training_config_adjust_list = ['device']
training_config = {
    'pad_id': 0,
    'batch_size': 2048, 
    'lr': 1e-3,
    'weight_decay': 0.0001,
    'epoch': 20, 
    'log_steps': 5, 
    'prediction_out_file': 'tmp/nb/predictions/prediction.pkl',
    'data_file': 'data/Newblocks/nb_train.pkl'

}


args_type = {
    'block1': 'type0',     
    'block2': 'type0',      
    'block3': 'type0',     
    'block4': 'type0',     
    'block5': 'type0',     
    'robot': 'type1'
}

label2word = {
    0: 13,# ready
    1: 3, # table
    2: 2, # on
    3: 40, # holding
    4: 16, #'nothing'
}


def hash_param(param, id2token):
    if isinstance(param, int):
        if param == 0:
            return 0
        token = id2token[param]
        token_t = args_type[token]
        return {'type0': 1, 'type1': 5}[token_t]
    elif isinstance(param, str):
        token = param
        token_t = args_type[token]
        return {'type0': 1, 'type1': 5}[token_t]
    elif isinstance(param, list):
        return [hash_param(i, id2token) for i in param]
    else:
        raise ValueError()
    

def get_param_type(param):
    if isinstance(param, int):
        return param

    if isinstance(param, list):
        if isinstance(param[0], int):
            return {1: 0, 2: 1, 5: 2, 6: 3}[sum(param)]
        elif isinstance(param[0], list):
            return [get_param_type(i) for i in param]

    else:
        raise ValueError()
    
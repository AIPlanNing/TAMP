model_config_adjust_list = ['vocab_size']

model_config = {
    'embd_hid': 128, 
    'hidden': 128,
    'dropout': 0.1,
    'encoder_num': 2,
    'encoder_name': 'gru',
    'embedding_from_wv': True,
    'wv_model_file': 'tmp/wv/wv.model',
    'embedding_freeze': True, 
    'state_encoder_type': 'rnn', # rnn | avg
    'temperature': 1,
}


training_config_adjust_list = ['device']
training_config = {
    'pad_id': 0,
    'batch_size': 2048, 
    'lr': 1e-3,
    'weight_decay': 0.0001,
    'epoch': 20, 
    'log_steps': 5, 
    'prediction_out_file': 'tmp/prediction.pkl'

}


args_type = {
    'block1': 'type0',     
    'block2': 'type0',      
    'block3': 'type0',     
    'block4': 'type0',     
    'block5': 'type0',     
    'robot': 'type1'
}
from collections import defaultdict
import os

import torch
from model.NTM import NTM
from dataset.dataset import Data
from config.nb_config import args_type as nb_args_type
from config.mc_config import args_type as mc_args_type
from dataset.tokenizer import BaseTokenizer
# from model.Trainer import NTMTrainer
from Rounder.Rounder import McRecNTMRounder, NTMRounder, RecNTMRounder

from utils import check_task
from plan_generator.generator import eval_action_model


import argparse



def get_dataset(raw_data_file, binary_file, cache_dir, from_scratch, limits=-1):
    if from_scratch  or not os.path.exists(binary_file):
        if not raw_data_file:
            raise ValueError(f'raw_data_file {raw_data_file} does not exist.')
        dataset = Data(
            raw_data_file, 
            cache_dir, 
            args_type=nb_args_type, 
            force=False,
            tokenizer=BaseTokenizer(),
            limits=limits
        )
        dataset_save_dir, _ = os.path.split(binary_file)
        os.makedirs(dataset_save_dir, exist_ok=True)
        dataset.save(binary_file)
        
    elif binary_file:
        dataset = Data.load(binary_file)
    
    return dataset







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rounds', type=int, default=20)
    parser.add_argument('--task', type=str, default='nb')

    #parser.add_argument('--train_data_file', type=str, default='data/Newblocks/nb_train.pkl')
    #parser.add_argument('--eval_data_file', type=str, default='data/Newblocks/nb_dev.pkl')
    parser.add_argument('--dataset_from_scratch', action='store_true')
    parser.add_argument('--data_input_num', default=-1, type=int, help='number of input plans')
    #parser.add_argument('--train_binary_file', type=str, default='tmp/nb/dataset/dataset.pkl')
    #parser.add_argument('--eval_binary_file', type=str, default='tmp/nb/dataset/dev_dataset.pkl')

    parser.add_argument('--num_topics', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='eevae')
    parser.add_argument('--num_param_types', type=int, default=4)
    parser.add_argument('--embd_size', type=int, default=64)
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--enc_type', type=str, default='gru')

    parser.add_argument('--train_bs', type=int, default=32)
    parser.add_argument('--predict_bs', type=int, default=1024)
    parser.add_argument('--lr', default=1e-2)
    parser.add_argument('--f_lr', default=1e-4)
    parser.add_argument('--learn_bs', default=20)
    parser.add_argument('--num_epoch', default=3, type=int)
    parser.add_argument('--log_steps', default=500)
    parser.add_argument('--save_steps', default=2)

    parser.add_argument('--cache_root', default='tmp')
    parser.add_argument('--soln_prefix', default='Soln')

    parser.add_argument('--htnml', default='HTNML/HTNML')
    parser.add_argument('--profile', default='HTNML/profile')
    parser.add_argument('--ratio', type=float, default=0.7)



    args = parser.parse_args()
    task2dir = {
        'nb': 'Newblocks',
        'mc': 'Minecraft_base'
    }
    args.train_data_file = os.path.join('data', task2dir[args.task], f'{args.task}_train.pkl')
    args.eval_data_file = os.path.join('data', task2dir[args.task], f'{args.task}_dev.pkl')
    args.train_binary_file = os.path.join(args.cache_root, args.task, 'dataset', 'dataset.pkl')
    args.eval_binary_file = os.path.join(args.cache_root, args.task, 'dataset', 'dev_dataset.pkl')
    

    check_task(args.train_data_file, args.task)

    cache_dir = os.path.join(args.cache_root, args.task)
    os.makedirs(cache_dir, exist_ok=True)

    
    train_dataset = get_dataset(args.train_data_file, args.train_binary_file, cache_dir, args.dataset_from_scratch, limits=args.data_input_num)
    eval_dataset = get_dataset(args.eval_data_file, args.eval_binary_file, cache_dir, args.dataset_from_scratch)
    eval_dataset.dictionary = train_dataset.dictionary


    
    stop_words = ['a', 'of', 'and', 'then', 'the', 'is', 'there', 'are']
    args.vocab_size = len(train_dataset.dictionary.token2id)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = NTM(
        vocab_size=args.vocab_size,
        num_topics=args.num_topics,
        num_param_types=args.num_param_types,
        num_action=len(train_dataset.actions),
        embd_size=args.embd_size,
        model_type=args.model_type,
        enc_type=args.enc_type,
        task=args.task,
        hid_size=args.hid_size
    )

    rounder = RecNTMRounder(
        model,
        train_dataset,
        args.num_rounds, 
        args.task,
        args.vocab_size,
        args.num_topics,
        len(train_dataset.actions),
        train_bs=args.train_bs,
        predict_bs=args.predict_bs,
        lr=args.lr,
        f_lr=args.f_lr,
        num_epoch=args.num_epoch,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        device=args.device,
        cache_root=args.cache_root,
        soln_prefix=args.soln_prefix,
        htnml=args.htnml,
        profile=args.profile,
        learn_bs=args.learn_bs,
        eval_dataset=eval_dataset,
        train_data_file=args.train_data_file,
        eval_data_file=args.eval_data_file,
        stop_words=stop_words
    )

    rounder(rebuild=False, ratio=args.ratio)
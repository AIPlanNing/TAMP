from collections import defaultdict
import glob
import os
import time

import torch
from model.NTM import NTM
from dataset.dataset import Data
from config.nb_config import args_type as nb_args_type
from dataset.tokenizer import BaseTokenizer
from plan_generator.utils import error_rate
from.Planner import Result, Plan
# from model.Trainer import NTMTrainer
from .Trainer import BkTrainer, McTrainer, NTMTrainer
from .Predictor import BkPredictor, McPredictor, NTMPredictor
from .SolutionGenerator import SolutionGenerator
from .Updater import RecUpdater, Updater
from .Learner import Learner
from utils import check_task
from .utils import LossScheduler, eval_action_model
from torch.utils.tensorboard.writer import SummaryWriter

import pickle as pkl
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#deal with pytorch tf conflict






class Rounder():
    def __init__(self, num_rounds):
        
        self.num_rounds = num_rounds
        pass

    def __call__(self):
        raise NotImplementedError()


    def train(self):
        pass

    def predict(self):
        pass

    def generate_solutions(self):
        pass

    def learn(self):
        pass


class NTMRounder(Rounder):

    

    def __init__(self, model, train_dataset, num_rounds, task, vocab_size, num_topics, num_actions, **kwargs):
        super().__init__(num_rounds)
        
        
        self.model = model
        self.train_dataset: Data = train_dataset

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_actions = num_actions
        self.task = task

        self.train_bs = kwargs.get('train_bs', 32)
        self.predict_bs = kwargs.get('predict_bs', 1024)
        self.lr = kwargs.get('lr', 1e-2)
        self.f_lr = kwargs.get('f_lr', self.lr)
        self.num_epoch = kwargs.get('num_epoch', 20)
        self.log_steps = kwargs.get('log_steps', 500)
        self.save_steps = kwargs.get('save_steps', 2)
        self.device = kwargs.get('device', 'cpu')
        self.model.to(self.device)

        self.cache_root = kwargs.get('cache_root', 'tmp')
        self.soln_prefix = kwargs.get('soln_prefix', 'Soln')

        self.eval_dataset = kwargs.get('eval_dataset')
        self.htnml_file = kwargs.get('htnml')
        self.profile = kwargs.get('profile')
        self.learn_bs = kwargs.get('learn_bs')

        self.stop_words = kwargs.get('stop_words')
        
        self.trainer = NTMTrainer(self.model, self.train_dataset, self.eval_dataset, self.task)
        self.predictor = NTMPredictor(self.model, self.train_dataset, self.eval_dataset, self.stop_words)
        self.soln_generator = SolutionGenerator(self.train_dataset.args_type, self.soln_prefix)
        self.learner = Learner(self.htnml_file, self.profile, self.soln_prefix, self.learn_bs)


        self.eval_data_file = kwargs.get('eval_data_file')
        self.train_data_file = kwargs.get('train_data_file')



    def __call__(self, rebuild=False, **kwargs):
        ratio = kwargs.get('ratio', 0.7)
        tb_writer = SummaryWriter(f'{self.cache_root}/tb/{self.task}/{time.ctime()}')
        for current_round in range(self.num_rounds):
            


            logger.info(f'round{current_round}: start training...')
            self.train(current_round, rebuild)

            logger.info(f'round{current_round}: start predicting train dataset...')
            #predictions, predictions_dev = self.predict(current_round, 'pre', rebuild)
            predictions = self.predict_train(current_round, 'raw', rebuild)
            predict_rs = self.train_dataset.eval()
            logger.info(f'round{current_round}: predict_rs {predict_rs}')
            tb_writer.add_scalar('predict_rs', predict_rs, current_round)
            tb_writer.add_scalar('rand_score', predict_rs, 2*current_round)
            
            logger.info(f'round{current_round}: start predicting eval dataset...')
            eval_predictions = self.predict_eval(current_round, 'raw', rebuild)

            logger.info(f'round{current_round}: start generating solutions train...')
            self.generate_solutions(current_round, False, dev=False, raw_data_file=self.train_data_file, predictions=predictions, rebuild=rebuild)
            #self.generate_solutions(current_round, False, raw_data_file=self.eval_data_file, predictions=predictions_dev, dev=True, rebuild=rebuild)
            
            logger.info(f'round{current_round}: start generating solutions eval...')
            eval_plans = self.generate_solutions(current_round, False, dev=True, raw_data_file=self.eval_data_file, predictions=eval_predictions, rebuild=rebuild, is_eval=True)

            logger.info(f'round{current_round}: start learning...')
            self.learn(current_round, rebuild)
            
            logger.info(f'round{current_round}: updating dataset...')
            actions = self.update_dataset(self.train_dataset, current_round, rebuild, ratio=ratio)
            er, rr = eval_action_model(actions, eval_plans)
            tb_writer.add_scalar('error_rate', er, current_round)
            tb_writer.add_scalar('redundancy_rate', rr, current_round)
            logger.info(f'round{current_round}: er {er}, rr {rr}')
            #eval error rate redundancy rate

            modify_rs = self.train_dataset.eval()
            logger.info(f'round{current_round}: modify_rs {modify_rs}')
            logger.info(f'round{current_round}: {actions}')
            tb_writer.add_scalar('modify_rs', modify_rs, current_round)
            tb_writer.add_scalar('rand_score', modify_rs, 2*current_round+1)
            tb_writer.add_text('actions', actions.__repr__(), current_round)
        tb_writer.close()
    def update_model(self, model_file):
        self.model = torch.load(model_file)
        self.trainer.model = self.model
        self.predictor.model = self.model

    def update_dataset_(self, train_dataset_file=None, eval_dataset_file=None, dev=False):
        if train_dataset_file:
            self.train_dataset = Data.load(train_dataset_file)
            self.trainer.train_dataset = self.train_dataset
            self.predictor.train_dataset = self.train_dataset
        if eval_dataset_file:
            self.eval_dataset = Data.load(eval_dataset_file)
            self.trainer.eval_dataset = self.eval_dataset
            self.predictor.eval_dataset = self.eval_dataset
        
        
    

    def train(self, current_round, rebuild=False, **kwargs):
        use_label = False if current_round == 0 else True
        #use_siam = False
        use_param_type = True if current_round == 0 else False
        self._train(use_label, use_param_type, current_round, rebuild, **kwargs)

    def _train(self, use_label, use_param_type, current_round, rebuild=False, **kwargs):

        path = self.get_cache_path('model', current_round)
        log_file = self.get_log_file('model', current_round, False)
        if not rebuild and os.path.exists(path):
            self.update_model(path)
            return

        
        if current_round == 0:
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
        else:
            self.update_model(self.get_cache_path('model', 0))
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [0.0, 0.0, 0.0, 5.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
        self.trainer.train(
            batch_size=self.train_bs,
            lr=self.lr,
            f_lr=self.f_lr,
            num_epoch=self.num_epoch,
            use_label=use_label,
            use_param_type=use_param_type,
            log_steps=self.log_steps,
            save_steps=self.save_steps,
            log_file=log_file,
            loss_scheduler=loss_schedular,
            **kwargs
        )
        
        
        self.trainer.save_model(path)
        # return model

    def predict_eval(self, current_round, key='pre', rebuild=False, **kwargs):
        #TODO
        by_head = False if current_round==0 else True
        kwargs['by_head'] = by_head
        dataset_path = self.get_cache_path('predict_dataset', current_round, True)
        
        eval_path = self.get_cache_path('prediction', current_round, True)
        #eval_path = eval_path + '_eval'
        # eval_path = self.get_cache_path('prediction', current_round, True)
        if not rebuild and os.path.exists(eval_path) and os.path.exists(dataset_path):
            #train_f, eval_f = open(train_path, 'rb'), open(eval_path, 'rb')
            eval_f = open(eval_path, 'rb')
            eval_result = pkl.load(eval_f)
            eval_f.close()
            self.update_dataset_(eval_dataset_file=dataset_path)
            # eval_f.close()
        else:
            #train_result, dev_result = self.predictor.predict(self.predict_bs, train_path, eval_path, key)
            eval_result = self.predictor.predict_eval(self.predict_bs, eval_path, key, **kwargs)
            self.predictor.eval_dataset.save(dataset_path)
        return eval_result

    def predict_train(self, current_round, key='pre', rebuild=False, **kwargs):
        by_head = False if current_round==0 else True
        kwargs['by_head'] = by_head
        dataset_path = self.get_cache_path('predict_dataset', current_round, False)
        train_path = self.get_cache_path('prediction', current_round, False)
        # eval_path = self.get_cache_path('prediction', current_round, True)
        if not rebuild and os.path.exists(train_path) and os.path.exists(dataset_path):
            #train_f, eval_f = open(train_path, 'rb'), open(eval_path, 'rb')
            train_f = open(train_path, 'rb')
            train_result = pkl.load(train_f)
            train_f.close()
            self.update_dataset_(train_dataset_file=dataset_path)
            # eval_f.close()
        else:
            #train_result, dev_result = self.predictor.predict(self.predict_bs, train_path, eval_path, key)
            train_result = self.predictor.predict_train(self.predict_bs, train_path, key, **kwargs)
            self.predictor.train_dataset.save(dataset_path)
        return train_result

    def predict(self, current_round, key='pre', rebuild=False):
        
        train_path = self.get_cache_path('prediction', current_round, False)
        eval_path = self.get_cache_path('prediction', current_round, True)
        if not rebuild and os.path.exists(train_path) and os.path.exists(eval_path):
            train_f, eval_f = open(train_path, 'rb'), open(eval_path, 'rb')
            train_result, dev_result = pkl.load(train_f), pkl.load(eval_f)
            train_f.close()
            eval_f.close()
        else:
            train_result, dev_result = self.predictor.predict(self.predict_bs, train_path, eval_path, key)

        return train_result, dev_result


    def generate_solutions(self, current_round, use_gold_params=False, dev=False, rebuild=False, **kwargs):
        #kwargs["raw_data"] = self.train_dataset.raw
        soln_dir = self.get_cache_path('solution', current_round, dev)
        if not rebuild and os.path.exists(soln_dir):
            plans = []
            if dev:
                for file in glob.glob(os.path.join(soln_dir, 'Soln*')):
                    plans.append(Plan.parse(file))
            return plans
        ret = self.soln_generator.generate(soln_dir, use_gold_params, **kwargs)
        return ret

    def learn(self, current_round, rebuild=False):
        solution_dir = self.get_cache_path('solution', current_round)
        results_dir = self.get_cache_path('result', current_round)
        if os.path.exists(results_dir) and glob.glob(os.path.join(results_dir, 'Res*')):
            return
        self.learner.learn(solution_dir, results_dir)

    

    def update_dataset(self, current_round):
        soln_dir = self.get_cache_path('solution', current_round)
        res_dir = self.get_cache_path('result', current_round)
        dataset_save_path = self.get_cache_path('dataset', current_round+1)
        dev_soln_dir = self.get_cache_path('solution', current_round, dev=True)
        specific_res_file, _ = eval_action_model(res_dir, dev_soln_dir, verbose=True)
        self.train_dataset.rebuild_data(soln_dir, res_dir, specific_res_file)
        self.train_dataset.save(dataset_save_path)
        return 

    def get_cache_path(self, key=None, current_round=None, dev=False):
        '''
        返回cache的目录，给定key则返回对应key(model, dataset, solution...)的目录
        否则给定cache根目录，给定current_round则返回对应该round的文件地址
        cache结构为：\n
        -cache_root\n
            -task\n
                -model\n
                -solution\n
                -dataset\n
                -result\n
                -prediction\n
                -log\n
        :param key(str): None, 'model', 'solution', 'dataset', 'result', 'prediction', 'log', 'predict_dataset'
        :param current_round(int): 当前为第几个round，大于等于0时会返回具体文件地址
        '''
        prefix = os.path.join(self.cache_root, self.task)
        if key not in [None, 'model', 'solution', 'dataset', 'result', 'prediction', 'log', 'predict_dataset', 'eval_solution']:
            raise NotImplementedError()
        if key is None:
            return prefix
        if current_round < 0 or current_round is None:
            return os.path.join(prefix, key)
        path = os.path.join(prefix, key, f'{key}_round{current_round}')
        if dev:
            path += '_dev'
        return path


    def get_log_file(self, key=None, current_round=None, dev=False):
        '''
        返回cache的目录，给定key则返回对应key(model, dataset, solution...)的目录
        否则给定cache根目录，给定current_round则返回对应该round的文件地址
        cache结构为：\n
        -cache_root\n
            -task\n
                -model\n
                -solution\n
                -dataset\n
                -result\n
                -prediction\n
                -log\n
        :param key(str): None, 'model', 'solution', 'dataset', 'result', 'prediction', 'log'
        :param current_round(int): 当前为第几个round，大于等于0时会返回具体文件地址
        '''
        prefix = os.path.join(self.cache_root, self.task, 'log')
        if key not in [None, 'model', 'solution', 'dataset', 'result', 'prediction', 'log']:
            raise NotImplementedError()
        if key is None:
            os.makedirs(prefix, exist_ok=True)
            return prefix
        os.makedirs(os.path.join(prefix, key), exist_ok=True)
        if current_round < 0 or current_round is None:
            
            return os.path.join(prefix, key)
        path = os.path.join(prefix, key, f'{key}_round{current_round}')
        if dev:
            path += '_dev'
        return path+'.log'
    
    def get_label_class_map(self, dataset):
        _map = None
        
        _data = [i.predicate for i in dataset.psuedo.values()]
        _map = {}
        for idx, item in enumerate(list(set(_data))):
            _map[item] = idx
        return _map

class RecNTMRounder(NTMRounder):
    def __init__(self, model, train_dataset, num_rounds, task, vocab_size, num_topics, num_actions, **kwargs):
        super().__init__(model, train_dataset, num_rounds, task, vocab_size, num_topics, num_actions, **kwargs)
        self.updater = RecUpdater(self.train_dataset, [], [], self.device)



    def update_dataset(self, dataset: Data, current_round, rebuild, **kwargs):
        ratio = kwargs.get('ratio', 0.7)
        dataset_save_path = self.get_cache_path('dataset', current_round+1)
        dir, _ = os.path.split(dataset_save_path)
        res_save_file = os.path.join(dir, f'result_round{current_round}')
        if not rebuild and os.path.exists(dataset_save_path):
            self.update_dataset_(dataset_save_path)
            actions = Result.parse(res_save_file).ams
            return actions
        results_dir = self.get_cache_path('result', current_round, False)
        soln_dir = self.get_cache_path('solution', current_round, False)
        res_file_list = glob.glob(os.path.join(results_dir, 'Result*'))
        plan_file_list = glob.glob(os.path.join(soln_dir, 'Soln*'))
        results = [Result.parse(i) for i in res_file_list]
        plans = [Plan.parse(i) for i in plan_file_list]
        self.updater.rec = (results, plans)

        pred_file = self.get_cache_path('prediction', current_round, False)

        record, actions = self.updater.update(dataset, prediction_file=pred_file, ratio=ratio)
        
        result = Result.from_ams(actions)
        result.output(res_save_file)
        
        dataset.save(dataset_save_path)
        return actions


class McRecNTMRounder(RecNTMRounder):
    def __init__(self, model, train_dataset, num_rounds, task, vocab_size, num_topics, num_actions, **kwargs):
        super().__init__(model, train_dataset, num_rounds, task, vocab_size, num_topics, num_actions, **kwargs)
        
        self.trainer = McTrainer(self.model, self.train_dataset, self.eval_dataset, self.task)
        self.predictor = McPredictor(self.model, self.train_dataset, self.eval_dataset, self.stop_words)


class BkRecNTMRounder(RecNTMRounder):
    def __init__(self, model, train_dataset, num_rounds, task, vocab_size, num_topics, num_actions, **kwargs):
        super().__init__(model, train_dataset, num_rounds, task, vocab_size, num_topics, num_actions, **kwargs)
        self.trainer = BkTrainer(self.model, self.train_dataset, self.eval_dataset, self.task)
        self.predictor = BkPredictor(self.model, self.train_dataset, self.eval_dataset, self.stop_words)


class NbBaseLineRounder(RecNTMRounder):
    def _train(self, use_label, use_param_type, current_round, rebuild=False, **kwargs):

        path = self.get_cache_path('model', current_round)
        log_file = self.get_log_file('model', current_round, False)
        if not rebuild and os.path.exists(path):
            self.update_model(path)
            return

        
        if current_round == 0:
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
            self.trainer.save_model(path)
            return
        else:
            self.update_model(self.get_cache_path('model', 0))
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [0.0, 0.0, 0.0, 5.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
        self.trainer.train(
            batch_size=self.train_bs,
            lr=self.lr,
            f_lr=self.f_lr,
            num_epoch=self.num_epoch,
            use_label=use_label,
            use_param_type=use_param_type,
            log_steps=self.log_steps,
            save_steps=self.save_steps,
            log_file=log_file,
            loss_scheduler=loss_schedular,
            **kwargs
        )
        self.trainer.save_model(path)


class McBaseLineRounder(McRecNTMRounder):
    def _train(self, use_label, use_param_type, current_round, rebuild=False, **kwargs):

        path = self.get_cache_path('model', current_round)
        log_file = self.get_log_file('model', current_round, False)
        if not rebuild and os.path.exists(path):
            self.update_model(path)
            return

        
        if current_round == 0:
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
            self.trainer.save_model(path)
            return
        else:
            self.update_model(self.get_cache_path('model', 0))
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [0.0, 0.0, 0.0, 5.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
        self.trainer.train(
            batch_size=self.train_bs,
            lr=self.lr,
            f_lr=self.f_lr,
            num_epoch=self.num_epoch,
            use_label=use_label,
            use_param_type=use_param_type,
            log_steps=self.log_steps,
            save_steps=self.save_steps,
            log_file=log_file,
            loss_scheduler=loss_schedular,
            **kwargs
        )
        self.trainer.save_model(path)

class BKBaseLineRounder(BkRecNTMRounder):
    def _train(self, use_label, use_param_type, current_round, rebuild=False, **kwargs):

        path = self.get_cache_path('model', current_round)
        log_file = self.get_log_file('model', current_round, False)
        if not rebuild and os.path.exists(path):
            self.update_model(path)
            return

        
        if current_round == 0:
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
            self.trainer.save_model(path)
            return
        else:
            self.update_model(self.get_cache_path('model', 0))
            loss_schedular = LossScheduler(
                ['prop_recon', 'prop_KLD', 'param_predict', 'label_predict', 'param_sort', 'total_loss'],
                [0.0, 0.0, 0.0, 5.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                0,
                10
            )
        self.trainer.train(
            batch_size=self.train_bs,
            lr=self.lr,
            f_lr=self.f_lr,
            num_epoch=self.num_epoch,
            use_label=use_label,
            use_param_type=use_param_type,
            log_steps=self.log_steps,
            save_steps=self.save_steps,
            log_file=log_file,
            loss_scheduler=loss_schedular,
            **kwargs
        )
        self.trainer.save_model(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rounds', type=int, default=5)
    parser.add_argument('task', type=str, default='nb')

    parser.add_argument('--train_data_file', type=str, default='data/Newblocks/nb_train.pkl')
    parser.add_argument('--eval_data_file', type=str, default='data/Newblocks/nb_dev.pkl')
    parser.add_argument('--dataset_from_scratch', action='store_true')
    parser.add_argument('--dataset_binary_file', type=str, default='tmp/nb/dataset/dataset.pkl')

    parser.add_argument('--num_topics', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='evae')
    parser.add_argument('--embd_size', type=int, default=64)
    parser.add_argument('--enc_type', type=str, default='gru')

    parser.add_argument('--train_bs', type=int, default=32)
    parser.add_argument('--predict_bs', type=int, default=1024)
    parser.add_argument('--lr', default=1e-2)
    parser.add_argument('--learn_bs', default=20)
    parser.add_argument('--num_epoch', default=20)
    parser.add_argument('--log_steps', default=500)
    parser.add_argument('--save_steps', default=2)

    parser.add_argument('--cache_root', default='tmp')
    parser.add_argument('--soln_prefix', 'Soln')

    parser.add_argument('--htnml', default='HTNML/HTNML')
    parser.add_argument('--profile', default='HTNML/profile')



    args = parser.parse_args()
    check_task(args.train_data_file, args.task)

    cache_dir = os.path.join(args.cache_dir, args.task)
    os.makedirs(cache_dir, exist_ok=True)

    if args.dataset_from_scratch and args.train_data_file:
        train_dataset = Data(
            args.train_data_file, 
            cache_dir, 
            args_type=nb_args_type, 
            force=False,
            tokenizer=BaseTokenizer()
        )
        dataset_save_dir, _ = os.path.split(args.dataset_binary_file)
        os.makedirs(dataset_save_dir, exist_ok=True)
        train_dataset.save(args.dataset_binary_file)
        
    elif args.dataset_binary_file:
        train_dataset = Data.load(args.dataset_binary_file)

    args.vocab_size = len(train_dataset.dictionary.token2id)
    args.device = torch.device('cuda' if torch.cuda.isavailable() else 'cpu')
    
    model = NTM(
        vocab_size=args.vocab_size,
        num_topics=args.num_topics,
        num_action=len(train_dataset.actions),
        embd_size=args.embd_size,
        model_type=args.model_type,
        enc_type=args.enc_type,
        task=args.task
    )

    rounder = NTMRounder(
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
        num_epoch=args.num_epoch,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        device=args.device,
        cache_root=args.cache_root,
        soln_prefix=args.soln_prefix,
        htnml=args.htnml,
        profile=args.profile,
        learn_bs=args.learn_bs
    )
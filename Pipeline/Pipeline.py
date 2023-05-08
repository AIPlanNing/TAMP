from functools import reduce
from Rounder.SolutionGenerator import SolutionGenerator
from Rounder.Learner import Learner
from Rounder.Trainer import Trainer
from Rounder.Predictor import NTMPredictor
from Rounder.Updater import RecUpdater
from baseline.Baseline import Baseline
from baseline.Baseline1 import BaselineDEP
from Rounder.utils import ActionModelModule
from model.models import ModelOutputForPlan
from plan_generator.Plan import ActionModel, Plan, Predicate, State
from plan_generator.Result import Result
from plan_generator.utils import error_rate, redundancy_rate
import torch
import glob
import os
from multiprocessing.pool import Pool
import pickle

TASK2RAW_DATA_FILE = {
    'nb': 'data/Newblocks/nb_train.pkl',
    'mc': 'data/Minecraft_base/mc_train.pkl',
    'bk': 'data/Baking/bk_train.pkl'
}
TASK2RAW_DATA_FILE_TEST = {
    'nb': 'data/Newblocks/nb_dev.pkl',
    'mc': 'data/Minecraft_base/mc_dev.pkl',
    'bk': 'data/Baking/bk_dev.pkl'
}

class Pipeline():

    def __init__(self, args_type, model: Baseline, task, raw_data_file, soln_out, res_out, learn_batch=1) -> None:
        self.task = task
        self.res_out = res_out
        self.soln_out = soln_out
        self.model = model
        self.args_type = args_type
        self.raw_data_file = raw_data_file
        self.learn_batch = learn_batch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.soln_generator = SolutionGenerator(args_type=args_type)
        self.Learner = Learner(learn_batch=learn_batch)
        

    def __call__(self, sents: dict):
        if os.path.exists(self.res_out):
            result = Result.parse(os.path.join(self.res_out, f'{self.task}_Result'))
            return {
                'result': result,
                'ams': result.ams
            }
        predictions = self.predict(sents)
        with open(f'{self.task}_DEP_predictions.pkl', 'wb') as f:
            pickle.dump(predictions, f)
        plans = self.generate(predictions)
        self.learn()
        results = []
        for file in glob.glob(os.path.join(self.res_out, 'Result.out*')):
            results.append(Result.parse(file))
        ams = self.get_best_am(results=results)
        result = Result.from_ams(ams)
        result.output(os.path.join(self.res_out, f'{self.task}_Result'))
        return {
            #'predictions': predictions,
            'result': result,
            'ams': ams
        }

    def predict(self, sents: dict):
        result = []
        for idx, sent in sents.items():
            pr, pa = self.model.parse_sentence(sent)
            data = {
                'id': idx,
                'predicate': pr,
                'extract_params': pa,
                'gold_params': pa,
                'label': None,
                'sentence': sent,
                'is_goal': None,
                'args_type': self.args_type
            }
            result.append(ModelOutputForPlan(data))
        result.sort(key=lambda x: x.id)
        return result


    def generate(self, predictions):
        return self.soln_generator.generate(self.soln_out, False, predictions=predictions, raw_data_file=self.raw_data_file)
    

    def learn(self):
        self.Learner.learn(self.soln_out, self.res_out)
    
    def get_best_am(self, results):
        action_names = reduce(lambda x, y: x+y, [list(j.ams.keys()) for j in results])
        action_names = list(set(action_names))


        updater = RecUpdater(None, results, [], self.device)
        ams = {}
        ret = updater._get_g_m_full(updater.rec.actions, lr=1e-1, num_steps=25000, ratio=0.5)
        for action_name in updater.rec.actions:
            precondition = ret[f'{action_name}_precondition']['g']
            precondition = updater.rec.vec2predicates(precondition, action_name, 'precondition')
            del_list = ret[f'{action_name}_del']['g']
            del_list = updater.rec.vec2predicates(del_list, action_name, 'del')
            add_list = ret[f'{action_name}_add']['g']
            add_list = updater.rec.vec2predicates(add_list, action_name, 'add')
            am = ActionModel.init_from(action_name, updater.rec.results[5].ams[action_name.lower()].parameters, precondition, del_list, add_list)

            ams[action_name.lower()] = am 
        return ams
    
    def eval_(self, ams, plans):
        lst_error_rate = [error_rate(ams, i) for i in plans]
        lst_redundancy_rate = [redundancy_rate(ams, i) for i in plans]
        return sum(lst_error_rate) / len(lst_error_rate), sum(lst_redundancy_rate) / len(lst_redundancy_rate)


    def eval(self, ams, sents):
        predictions = self.predict(sents)
        with open(f'{self.task}_dev_DEP_predictions.pkl', 'wb') as f:
            pickle.dump(predictions, f)
        plans = self.soln_generator.generate(self.soln_out+'dev', False, predictions=predictions, raw_data_file=TASK2RAW_DATA_FILE_TEST[self.task])
        er, rr = self.eval_(ams, plans)
        eval_save_path = os.path.join(self.soln_out, 'eval.txt')
        with open(eval_save_path, 'w') as f:
            f.write(f'error_rate: {er}\nredundancy_rate: {rr}')
        
        return er, rr

class PipelineDEP(Pipeline):

    def __init__(self, args_type, task, raw_data_file=None, learn_batch=1) -> None:
        model = BaselineDEP(task, args_type)
        soln_out = os.path.join('tmp', task, 'baselineDEP', 'solution') #TODO
        res_out = os.path.join('tmp', task, 'baselineDEP', 'result')
        if not raw_data_file:
            raw_data_file = TASK2RAW_DATA_FILE[task]
        super().__init__(args_type, model, task, raw_data_file, soln_out, res_out, learn_batch=learn_batch)
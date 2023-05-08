from copy import deepcopy
from collections import defaultdict
from config import mc_config, nb_config, bk_config
from dataset.tokenizer import BaseTokenizer



class Baseline():
    TASK2FUNC_HASH_PARAM = {
        'nb': nb_config.hash_param,
        'mc': mc_config.hash_param,
        'bk': bk_config.hash_param
    }
    TASK2FUNC_GET_PARAM_TYPE = {
        'nb': nb_config.get_param_type,
        'mc': mc_config.get_param_type,
        'bk': bk_config.get_param_type
    }

    def __init__(self, task, args_type) -> None:
        assert isinstance(args_type, dict)
        assert task in ['nb', 'mc', 'bk']
        self.__task = task
        self.__predicates = set()
        self.__predicate2param_type = defaultdict(int)# predicate name-i
        self.__predicate_variants = defaultdict(int) # str to int
        self.__args_type = args_type
        self.tokenizer = BaseTokenizer()

    @property
    def task(self):
        return self.__task[:]

    @property
    def args_type(self):
        return deepcopy(self.__args_type)

    @property
    def predicates(self):
        return self.__predicates[:]

    @property
    def predicate2param_type(self):
        return deepcopy(self.__predicate2param_type)

    @property
    def predicate_variants(self):
        return deepcopy(self.__predicate_variants)

    def commit_predicate(self, predicate: str, param: list, id2token=None):
        variants = self.__predicate_variants[predicate]
        pt = self.get_param_type(self.hash_param(param, id2token))
        if variants == 0:
            self.__predicate_variants[predicate] += 1
            self.__predicate2param_type[predicate] = pt
            self.__predicates.update([predicate])
            return predicate
        else:
            match = False
            for predicate_ext in [predicate] + [f'{predicate}-{i}' for i in range(1, variants)]:
                
                if pt == self.__predicate2param_type[predicate_ext]:
                    match = True
                    return predicate_ext

            if not match:
                self.__predicate_variants[predicate] += 1
                self.__predicate2param_type[f'{predicate}-{variants}'] = pt
                self.__predicates.update([f'{predicate}-{variants}'])
                return f'{predicate}-{variants}'

    def get_param_type(self, param):
        return self.TASK2FUNC_GET_PARAM_TYPE[self.task](param)
    
    def hash_param(self, param, id2token=None):
        return self.TASK2FUNC_HASH_PARAM[self.task](param, id2token)


    def extract_param(self, sent):
        ret = []
        tokens = self.tokenizer.tokenize(sent)
        for item in tokens:
            if item.lower() in self.args_type.keys():
                ret.append(item.lower())

        return ret

    def parse_sentence(self, sent):
        raise NotImplementedError()
        
        

    def __call__(self):
        
        raise NotImplementedError()
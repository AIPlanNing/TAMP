from abc import abstractmethod
import os
import glob

from plan_generator.Plan import ActionModel, State

class BaseLearner():
    def __init__(self) -> None:
        pass

    def learn(self):
        raise NotImplementedError()

class Learner(BaseLearner):
    def __init__(self, src='HTNML/HTNML', profile='HTNML/Profile', soln_prefix='Soln', learn_batch=20):
        self.soln_prefix = 'Soln'
        self.src = src
        self.learn_batch = learn_batch
        self.profile = profile
        self.write_profile()
        # profile = os.path.join(os.path.split(src)[0], 'Profile')

    
    def write_profile(self):
        with open(self.profile, 'w') as f:
            content = ['am1', str(self.learn_batch), '1.0', '1.0', 'am1/Maxsat.in', 'am1/Maxsat1.in', 'am1/Result.out']
            f.write('\n'.join(content))
    def learn(self, solution_dir=None, results_dir=None):
        self.write_profile()
        src_dir, src = os.path.split(self.src)
        
        if solution_dir is not None:
            # 清空目录
            file_to_be_removed = glob.glob(os.path.join(src_dir, 'am1', '*'))
            [os.remove(i) for i in file_to_be_removed]
            # 将soln文件放入目录
            os.system(f'cp {os.path.join(solution_dir, self.soln_prefix+"*")} {os.path.join(src_dir,  "am1")}')

        self._learn(src_dir, src, results_dir)
        '''os.chdir(src_dir)
        # 执行
        os.system(f'./{src} > /dev/null')
        os.chdir('..')
        os.makedirs(results_dir, exist_ok=True)
        os.system(f'mv {os.path.join(src_dir, "am1", "Result.out*")} {results_dir}')
        '''

    def _learn(self, src_dir=None, src=None, results_dir=None):
        _src_dir, _src = os.path.split(self.src)
        src_dir = src_dir if src_dir is not None else _src_dir
        src = src if src is not None else _src
        os.chdir(src_dir)
        os.system(f'./{src} > /dev/null')
        os.chdir('..')
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            os.system(f'mv {os.path.join(src_dir, "am1", "Result.out*")} {results_dir}')



class SimpleLearner(BaseLearner):
    def __init__(self) -> None:
        super().__init__()
        
    def learn(self, inp):
        '''
        :param inp: tuple(s_prior, action, s_post)
        '''
        assert len(inp) == 3
        assert isinstance(inp[0], State) and isinstance(inp[2], State)
        prior, action, post = inp
        assert isinstance(action[1], dict)
        name = action[0]
        para = action[1]
        precondition = prior.filter(action[1].keys())
        del_list = prior - post
        add_list = post - prior


        return ActionModel.init_from(name, para, precondition, del_list, add_list)
        

    

if __name__ == '__main__':
    learner = Learner()
    learner.learn()
    pass
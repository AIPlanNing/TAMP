import enum
from model.models import ModelOutputForPlan
import copy, re, os


class Param():

    def __init__(self, args_type: dict, **kwargs) -> None:
        param = kwargs.get('param')
        if not param:
            param = kwargs.get('params')
        if param:
            self.__type = {}
            for item in param:
                self.__type[item] = args_type[item]
        else:
            self.__type = copy.deepcopy(args_type)
        self.__param = list(self.__type.keys())
        self._add_index_int2type()


    def name_equal(self, p):
        return self.param == p.param

    def name_type_equal(self, p):
        return self.name_equal(p) and [self.type[i] for i in self.param] == [p.type[i] for i in p.param]

    def _add_index_int2type(self):
        for idx, item in enumerate(self.__param):
            self.__type[idx] = self.__type[item]


    def __repr__(self) -> str:
        return repr([(i, self.__type[i]) for i in self.__param])

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, index):
        return self.__param[index]

    def __len__(self):
        return len(self.__param)

    @property
    def param(self):
        return self.__param[:]

    @property
    def type(self):
        return self.__type


    def param_change_(self, src2tgt: dict):
        
        # self.__param = list(src2tgt.values())
        temp = {}
        for src in self.__param:
            temp[src2tgt[src]] = self.__type[src]

        self.__param = [src2tgt[i] for i in self.__param]
        '''for src, tgt in src2tgt.items():
            temp[tgt] = self.__type[src]'''
        self.__type = temp
        self._add_index_int2type()
        return self

    def param_change(self, src2tgt):
        n_p = copy.deepcopy(self)
        return n_p.param_change_(src2tgt)



class ActionModel():
    '''
    action schema
    '''
    def __init__(self, data: dict) -> None:
        self.name = data.get('name')
        self.parameters = data.get('parameters')
        self.precondition = data.get('precondition')
        self.del_list = data.get('del_list')
        self.add_list = data.get('add_list')


    @classmethod
    def init_from(cls, name, para, p, d, a):
        if isinstance(p, list):
            p = State.predicate_init(p)
        if isinstance(d, list):
            d = State.predicate_init(d)
        if isinstance(a, list):
            a = State.predicate_init(a)
        return cls(
            {
                'name': name,
                'parameters': para,
                'precondition': p,
                'del_list': d,
                'add_list': a
            }
        )



    def params_change(self, params):
        assert len(params) == len(self.parameters)
        ret = copy.deepcopy(self)
        ret.parameters = {}
        params_map = {}
        for p, np in zip(self.parameters, params):
            ret.parameters[np] = self.parameters[p]
            params_map[p] = np
        ret.add_list = ret.add_list.params_change(params_map)
        ret.del_list = ret.del_list.params_change(params_map)
        ret.precondition = ret.precondition.params_change(params_map)
        return ret
        
        


    def regular(self, key='ad'):
        '''
        修改action model使其符合action constraint
        '''
        n_s = copy.deepcopy(self)
        
        if key == 'p':# 修改precondition
            # 处理矛盾，即add list 与del list的交集
            # TODO
            # 删除与add list的交集
            n_s.precondition = self.precondition - (self.precondition & self.add_list)
            # 添加del list 与precondition的差集
            n_s.precondition = self.precondition | (self.del_list - self.precondition)
        elif key == 'ad' or key == 'da':
            # add list去除与precondition的交集
            n_s.add_list = self.add_list - (self.add_list & self.precondition)
            # del list去除与precondition的差集
            n_s.del_list = self.del_list - (self.del_list - self.precondition)

        return n_s
    @classmethod
    def parse_param(cls, stream):
        '''
        (?a -TYPE0 ?b -TYPE1 )
        '''
        params = {}
        pattern = re.compile(r'\?[a-zA-Z\s]+\-[a-zA-Z0-9]+\s')
        al = pattern.findall(stream)
        for item in al:
            item = item.replace('-', '')
            item = item.split(' ')
            params[item[0]] = item[1].lower()
        return params


    @classmethod
    def parse_precondition(cls, stream, args_type):
        pattern = re.compile(r'\([A-Z0-9]+[\s\?a-z]+\)')
        al = pattern.findall(stream)
        precondition = State.parse('\n'.join(al)+'\n', args_type=args_type)
        return precondition
        

    @classmethod
    def parse_effect(cls, stream, args_type):
        del_pattern = re.compile(r'not\([A-Z0-9]+[\s\?a-z]+\)')
        add_pattern = re.compile(r'\([A-Z0-9]+[\s\?a-z]+\)')
        del_al = del_pattern.findall(stream)
        del_al = [i.replace('not', '') for i in del_al]
        add_al = add_pattern.findall(stream)
        del_list = State.parse('\n'.join(del_al)+'\n', args_type=args_type)
        add_list = State.parse('\n'.join(add_al)+'\n', args_type=args_type)
        add_list = add_list - del_list
        return del_list, add_list

    @classmethod
    def parse(cls, stream):
        data = {}
        stream = stream.split('\n')
        name = ''
        param_stream = ''
        pre_stream = ''
        eff_stream = ''
        for item in stream:
            if ':action' in item:
                name = item.split(' ')[1]
            elif ':parameters' in item:
                param_stream = ' '.join(item.split(' ')[1:])
            elif ':precondition' in item:
                pre_stream = ' '.join(item.split(' ')[1:])
            elif ':effect' in item:
                eff_stream = ' '.join(item.split(' ')[1:])
        data['name'] = name
        data['parameters'] = cls.parse_param(param_stream)
        data['precondition'] = cls.parse_precondition(pre_stream, data['parameters'])
        data['del_list'], data['add_list'] = cls.parse_effect(eff_stream, data['parameters'])
        return cls(data)

    def __repr__(self):
        return f'({self.name} ' + ' '.join(self.parameters) + ')'

class Action(ActionModel):
    
    @classmethod
    def instantiate_from(cls, am: ActionModel, params_map):
        data = {'parameters': {}, 'name': am.name}
        for p, t in am.parameters.items():
            data['parameters'][params_map[p]] = t
        
        # precondition
        data['precondition'] = am.precondition.params_change(params_map)
        data['del_list'] = am.del_list.params_change(params_map)
        data['add_list'] = am.add_list.params_change(params_map)
        return cls(data)
        # del_list

        # add_list


class Predicate():
    '''
    由命题名称和参数组成
    '''
    def __init__(self, input_data: ModelOutputForPlan, use_gold_params=False) -> None:
        params = input_data.extract_params[:] if not use_gold_params else input_data.gold_params[:]
        args_type = input_data.args_type
        self.params = Param(args_type, params=params)
        self.name = input_data.predicate




    def params_change_(self, params_map):
        '''
        改变该谓词的参数(self.params)，在将action model实例化时用到
        :param params_map: 原参数到目标参数的映射，如{'?a': 'Block1'}
        '''
        if isinstance(params_map, list):
            _params_map = {}
            for i, j in zip(self.params, params_map):
                _params_map[i] = j
            params_map = _params_map
        # n_p = copy.deepcopy(self)
        self.params = self.params.param_change(params_map)
        # for idx, p in enumerate(self.params):
        #     self.params[idx] = params_map[p]
        return self


    def params_change(self, params_map):
        n_p = copy.deepcopy(self)
        return n_p.params_change_(params_map)


    def __hash__(self):
        return hash(' '.join([self.name] + self.params.param))

    def __eq__(self, o: object) -> bool:
        if not len(self.params) == len(o.params):
            return False
        if not o.name.lower() == self.name.lower():
            return False
        for i, j in zip(self.params, o.params):
            if not i.lower() == j.lower():
                return False
        
        return True

    def __str__(self) -> str:
        return f'({self.name} {self.params.param})'

    def __repr__(self) -> str:
        return f'({self.name} {self.params.param})'
            

    @classmethod
    def parse(cls, stream: str, args_type):
        '''
        stream looks like '(predicate param1 param2 ...)'
        '''
        stream = stream.replace('(', '')
        stream = stream.replace(')', '')
        
        return cls(ModelOutputForPlan({
            'extract_params': stream.split(' ')[1:],
            'predicate': stream.split(' ')[0],
            'id': None,
            'args_type': args_type
        }))


class State():
    '''
    由多个命题组成
    '''
    def __init__(self, info: dict) -> None:
        self.content = info['predicates'][:] # 命题列表
        self.plan_id = info['plan_id']
        self.state_id = info.get('state_id')
        self.is_goal = False
        self.precondition_of = None if self.is_goal else info.get('post_action')
        self.postcondition_of = None if self.is_goal else info.get('pre_action')
        self.is_init = False
        

    @classmethod
    def predicate_init(cls, predicate_list):
        assert isinstance(predicate_list, list)
        info = {
            'predicates': predicate_list,
            'plan_id': -1
        }
        obj = cls(info)
        return obj

    def filter(self, params):
        assert isinstance(params, list)
        ret = copy.deepcopy(self)
        ret.content = []
        for item in self.content:
            if len(set(item.params.param) - set(params)) == 0:
                ret.content.append(copy.deepcopy(item))
        return ret

                    

    def __getitem__(self, idx):
        return self.content[idx]

    def __setitem__(self, idx, value):
        if not isinstance(value, Predicate):
            raise ValueError(f'{value} is not a Predicate.')
        self.content[idx] = value

    def __len__(self):
        return len(self.content)

    def __call__(self, am: Action):
        '''
        将action作用到state上，返回下一个state
        有可能state缺少am.del_list的内容，
        有可能state存在am.add_list的内容
        :return {
            'next_state': next state,
            'to_be_add': 原状态需要增加的predicate列表（这些predicate在del_list出现
            但是没有在state中出现,
            'to_be_del': 原状态需要删除的predicate列表（这些predicate在add_list出现
            但是也在state中出现
        }
        
        '''
        
            
        n_s = copy.deepcopy(self)
        ret = {'next_state': self, 'to_be_add': [], 'to_be_del': []}
        
        '''
        action constraint 不满足，改del list\\add list还是改precondition
        plan constraint不满足（这个应该普遍存在），如何处理
        '''
        ret['to_be_add'] = am.precondition - n_s
        ret['to_be_del'] = am.add_list & n_s
        n_s = n_s | am.add_list
        n_s = n_s - am.del_list
        ret['next_state'] = n_s

                
        return ret

    def __and__(self, s):
        n_s = copy.deepcopy(self)
        n_s.content = []
        for pred in self:
            for comp_pred in s:
                if pred == comp_pred:
                    n_s.content.append(copy.deepcopy(pred))
                    break
        return n_s

    def __or__(self, s):
        n_s = copy.deepcopy(self)
        d = {}
        for item in n_s.content:
            key = ' '.join([item.name]+item.params.param)
            d[key] = copy.deepcopy(item)
        for item in s.content:
            key = ' '.join([item.name]+item.params.param)
            d[key] = copy.deepcopy(item)
        n_s.content = list(d.values())
        return n_s
    
    def __sub__(self, s):
        n_s = copy.deepcopy(self)
        for pred in self:
            for comp_pred in s:
                if pred == comp_pred: # pred in s
                    n_s.content.remove(pred)
                    break
        return n_s

    def __repr__(self) -> str:
        if self.is_empty():
            return 'empty state'
        return ','.join([i.__repr__() for i in self.content])

    def is_empty(self):
        if len(self) == 0:
            return True
        return False

    def params_change(self, params_map):
        '''
        返回参数改变后的State，将action model实例化时用到
        :param params_map 形如{'?a': 'Block1'}
        '''
        n_s = copy.deepcopy(self)
        n_s.content = [pred.params_change(params_map) for pred in self.content]
        return n_s



    @classmethod
    def parse(cls, stream, plan_id=-1, args_type=None):
        '''
        stream looks like '()\n()\n...()\n'
        '''
        if args_type is None:
            Warning('args_type is None.')
            args_type = {}
        info = {
            'predicates': [],
            'plan_id': plan_id
        }
        
        for p_stream in stream.split('\n')[:-1]:
            
            if re.match(r'[\s]+', p_stream) or not p_stream:
                continue
            assert not re.match(r'[\s]+', p_stream) and p_stream
            _pred = Predicate.parse(p_stream, args_type)
            if _pred:
                info['predicates'].append(_pred)

        return cls(info)

class Goal(State):
    def __init__(self, info: dict):
        self.content = info['predicates']
        self.plan_id = info['plan_id']
        self.is_goal = True

    def __len__(self):
        return len(self.content)

    


class Plan():
    def __init__(self, info: dict, args_type: dict={}) -> None:
        self.states = info['state_list'][:]
        self.plan_id = info['plan_id']
        self.actions = info['actions'][:] # action stream '(action param param)\n'
        self.init = None
        self.goal = info['goal']
        self.args_type = args_type


    @classmethod
    def init_from_action_states(cls, action, state, args_type):
        info = {
            'state_list': state,
            'actions': action,
            'plan_id': -1,
            'goal': None,
        }
        return cls(info, args_type)

    def __getitem__(self, idx):
        return self.states[idx]

    def __setitem__(self, idx, value):
        if not isinstance(value, State):
            raise ValueError(f'{value} is not type State.')
        self.states[idx] = value

    def __len__(self):
        return len(self.states)


    def output(self, path):
        '''
        将一个Plan对象写入到目标文件中。
        '''

        def _write_state(handler, state: State, prefix='observations'):
            handler.write(f'(:{prefix}\n')
            for predicate in state.content:
                handler.write(f'({predicate.name.lower()} ')
                handler.write(' '.join(predicate.params.param))
                handler.write(')\n')
            handler.write(')\n')


        dir, file_name = os.path.split(path)
        os.makedirs(dir, exist_ok=True)

        with open(path, 'w') as f:
            f.write('(solution\n')
            f.write('(:objects')
            for k, v in self.args_type.items():
                f.write(f' {k} - {v}')
            f.write(')\n') #object end
            _write_state(f, self.states[0], 'init')
            f.write('\n')
            for action, state in zip(self.actions[:], self.states[:]):
                _write_state(f, state)
                f.write(action)
                f.write('\n')
            _write_state(f, self.states[-1], 'goal')
            # _write_state(f, self.states[-1])
            # f.write('\n')
            # _write_state(f, self.goal, 'goal')
            

            f.write(')') # solution end
        
    @classmethod
    def parse(cls, soln_file):

        '''
        解析一个plan文件（一般在solution文件夹中）
        该plan文件需要以plan id结尾
        '''
        # read file
        plan_id = re.match(r'([a-zA-Z/]+)([0-9]+)', os.path.split(soln_file)[1]).groups()[1]
        plan_id = int(plan_id)
        state_stream = []
        action_stream = []
        current_process = ''
        state = ''
        args_type_steam = ''
        with open(soln_file, 'r') as f:
            while True:
                stream = f.readline()
                if not stream:
                    break
                if ':objects' in stream:
                    stream = stream.replace('(:objects ', '')
                    stream = stream.replace(')', '')
                    args_type_steam = stream
                if ':observations' in stream:
                    current_process = ':observations'
                    break
            while True:
                stream = f.readline()
                if not stream:
                    break
                
                if stream == '\n':
                    continue

                if current_process == 'action':
                    if ':observations' in stream or ':goal' in stream:
                        current_process = ':observations'
                        state = ''
                        continue
                    else:
                        action_stream.append(stream)
                elif current_process == ':observations':
                    if stream == ')\n':
                        current_process = 'action'
                        state_stream.append(state)
                        continue
                    else:
                        state += stream
        info = {}
        info['actions'] = action_stream[:-1]
        info['plan_id'] = plan_id
        info['state_list'] = []
        args_type = {}
        a_t_pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9\-]+\s\-\s[a-zA-Z][a-zA-Z0-9]+')
        args_type_group = a_t_pattern.findall(args_type_steam)
        for a_t in args_type_group:
            args_type[a_t.split(' - ')[0]] = a_t.split(' - ')[1]
        # print(args_type_group, args_type)
        # state
        for s_stream in state_stream:
            info['state_list'].append(State.parse(s_stream, plan_id, args_type))

        info['goal'] = info['state_list'][-1]

        return cls(info, args_type)
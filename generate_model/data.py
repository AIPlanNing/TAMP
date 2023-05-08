from torch.utils.data import DataLoader, Dataset
import argparse, logging
import pickle as pkl
import os
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from gensim.corpora import Dictionary

class TaskInfo:
    def __init__(self) -> None:
        self.predicates = []
        self.objects = []
        self.actions = []
        self.task = ""

class NBInfo(TaskInfo):
    def __init__(self) -> None:
        super().__init__()
        self.task = "nb"
        # TODO
        self.predicates = ["on", "ontable", "clear", "handempty", "holding"]
        self.objects = ["block1", "block2", "block3", "block4", "block5", "robot"]
        self.actions = ["pick-up", "put-down", "stack", "unstack"]
        

class MCInfo(TaskInfo):
    def __init__(self) -> None:
        self.task = "mc"
        # TODO
        self.predicates = []
        self.objects = []
        self.actions = []

class BKInfo(TaskInfo):
    def __init__(self) -> None:
        self.task = "bk"
        # TODO
        self.predicates = []
        self.objects = []
        self.actions = []
        

taskList = ["nb", "mc", "bk"]
def getTaskPath(task, partition):
    taskMap = {"nb": "Newblocks", "mc": "micecraft_base", "bk": "Baking"}
    path = {"train": "", "test": "", "dev": ""}
    for key in path.keys():
        path[key] = os.path.join("data", taskMap[task], f"{task}_{key}.pkl")
    return path[partition]


def checkArgs(args):
    if args.task not in taskList:
        raise ValueError("wrong task value.")
    if args.partition not in ["train", "test", "dev"]:
        raise ValueError("wrong partition value.")

class SymbolTokenizer():
    '''
    symbols includes action object and predicate
    '''
    def __init__(self, predicates: list, objects: list, actions: list) -> None:
        # assert(len(predicates) > 0, "SymbolTokenizer: no predicates")
        # assert(len(objects) > 0, "SymbolTOkenizer: no objects")

        self.nPredicate = len(predicates)
        self.nObject = len(objects) + 1
        self.pad = "<PAD>"
        self.padId = 0
        self.padObj = "<PADOBJ>"
        
        self.boa = "<BOA>" # begin of action
        self.eoa = "<EOA>" # end of action

        self.actions = actions + [self.boa, self.eoa]
        self.objects = objects + ["<PADOBJ>"]
        self.predicates = predicates

        self.id2token = [self.pad]
        self.token2id = {self.pad: self.padId} # 0: <pad>, 1: obj0... objm padobj pred1 m+1 pred2, action...
        for idx, item in enumerate(self.objects + self.predicates + self.actions):
            self.id2token.append(item)
            self.token2id[item] = idx + 1

    def decode(self, inp):
        # n, l, k
        ret = []
        for seq in inp:
            ret.append([[self.id2token[id] for id in act] for act in seq])
        return ret

    def tokenize(self, inp):
        if isinstance(inp, str):
            inp = inp.replace("(", "")
            inp = inp.replace(")", "")
            inp = inp.replace("\n", "")
            return inp.split(" ")
            #return [self.token2id[i] for i in inp.split(" ")]
        elif isinstance(inp, list):
            ret = []
            for item in inp:
                ret.append(self.tokenize(item))
            return ret
        else:
            raise ValueError("wrong input for tokenizer")

    def encode(self, inp, pad=3):
        '''
        :param pad(int): pad sequence with padobj to length, do not pad if pad is -1
        '''
        assert(isinstance(inp, str))
        tokens = self.tokenize(inp) + [self.padObj] * pad # FIXME: deal with differnt input
        return [self.token2id[i] for i in tokens][:pad]

class Tokenizer():
    def __init__(self) -> None:
        self.sent = PunktSentenceTokenizer()
        self.nlp = TreebankWordTokenizer()
        
    def tokenize(self, inp):
        assert(isinstance(inp, str))
        sents = self.sent.tokenize(inp)
        return [self.nlp.tokenize(i) for i in sents]

    
    def tokenize1(self, inp):
        if isinstance(inp, str):
            
            return self.nlp.tokenize(inp)
        elif isinstance(inp, list):
            ret = []
            for item in inp:
                ret.append(self.tokenize(item))
            return ret
        else:
            raise ValueError(f"wrong input for tokenizer, accept list or str. but got {type(inp)}")



class GernerateDataset(Dataset):
    def __init__(self, task, partition, cacheDir="tmp/action_generate_data", force=False):
        self.cacheDir = cacheDir
        self.force = force
        self.task = task
        self.partition = partition

        self.taskInfo: TaskInfo = {"nb": NBInfo, "mc": MCInfo, "bk": BKInfo}[task]()
        self.symbolTokenizer = SymbolTokenizer(self.taskInfo.predicates, self.taskInfo.objects, self.taskInfo.actions)


        self.tokenizer = Tokenizer()
        self.dictionary = Dictionary()
        self.padToken, self.padId = "<PAD>", 0

        path = getTaskPath(task, partition)
        with open(path, "rb") as f:
            self.data = pkl.load(f)
        self.planidToDataid = {} # find the id of data according planid
        self.goldProposition = [] # every item be like [[(p a, a), (p, a, a)]]
        self.description = [] # every item be like ["p a a. p a a. p a a.", "p a a"]
        self.goal_text = [] # be like "p a a. p a a."
        self.action = [] # be like ["(p a a)\n", "(p a a)\n"]

        self.__generate()
        self.__buildDict()
        
        
    def __buildDict(self):
        __filename = f"vocab_{self.taskInfo.task}"
        if not self.force and os.path.exists(os.path.join(self.cacheDir, __filename)):
            self.dictionary = Dictionary.load(os.path.join(self.cacheDir, __filename))
            return
        for item in self.description:
            for sent in item:
                _tokens = self.tokenizer.tokenize(sent)
                self.dictionary.add_documents(_tokens)

        _specialTokens = {
            self.padToken: self.padId
        }
        self.dictionary.patch_with_special_tokens(_specialTokens)
        for k, v in self.dictionary.token2id.items():
            self.dictionary.id2token[v] = k
        if self.partition == "train" and not os.path.exists(os.path.join(self.cacheDir, __filename)):

            os.makedirs(self.cacheDir, exist_ok=True)
            self.dictionary.save(os.path.join(self.cacheDir, __filename))

        

    def __generate(self):
        for dataid, plan in enumerate(self.data):
            self.planidToDataid[plan["id"]] = dataid

            # state traces
            propostions = plan["state"][1:]
            propostions = [plan["initial_state"]] + propostions
            self.goldProposition.append(propostions)

            # text traces
            self.description.append(plan["text_trace"])

            # goal text
            self.goal_text.append(plan["goal_text"])

            # action traces
            action = plan["plan"][:-1] 
            self.action.append(action)

        
    def __getitem__(self, index):
        data = {
            "planid": self.data[index]["id"], 
            "gold_proposition": self.goldProposition[index],
            "description": self.description[index],
            "goal_text": self.goal_text[index],
            "action": self.action[index]
        }
        return data
    
    def __len__(self):
        return len(self.data)

    def symbolEncode(self, inp, pad=3):
        assert(isinstance(inp, str))
        tokens = self.symbolTokenizer.encode(inp, pad=pad)
        return tokens

    def encode(self, inp):
        assert(isinstance(inp, str))
        tokens = self.tokenizer.tokenize(inp)
        return [[self.dictionary.token2id[tok] for tok in i] for i in tokens]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="nb", help="type of task, nb, mc, bk or all")
    parser.add_argument("--partition", type=str, default="train", help="train, test or dev")

    args = parser.parse_args()
    checkArgs(args)

    logger = logging.getLogger()
    logger.info("load dataset...")
    dataset = GernerateDataset(args.task, args.partition)
    print(dataset.encode("block1 is on table. block2 is under block1."))
    #print(dataset[0])
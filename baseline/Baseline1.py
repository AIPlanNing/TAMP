from .Baseline import Baseline
from stanfordcorenlp import StanfordCoreNLP


class BaselineCoreNLP(Baseline):
    def __init__(self, task, args_type) -> None:
        super().__init__(task, args_type)
        #self.__model = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
        self.model = StanfordCoreNLP('/home/chenhx/stanford-corenlp-4.3.2')



class BaselinePOS(BaselineCoreNLP):

    def __init__(self, task, args_type) -> None:
        super().__init__(task, args_type)

        
    

    def __call__(self):
        return super().__call__()



    def parse_sentence(self, sent):
        pos = self.model.pos_tag(sent)
        params = self.extract_param(sent)
        for token, tag in pos:
            if tag in ['VB', 'VBZ', 'VBD', 'VBG', 'VBN', 'VBP']:# deal with multi verb
                predicate = self.commit_predicate(token.lower(), params)
                break
        return predicate, params


    def pos(self, sent):
        return self.model.pos_tag(sent)


class BaselineDEP(BaselineCoreNLP):
    def __init__(self, task, args_type) -> None:
        super().__init__(task, args_type)
    
    def parse_sentence(self, sent):
        dep = self.model.dependency_parse(sent)# (tag, head, tail)
        params = self.extract_param(sent)
        tokens = self.model.word_tokenize(sent)
        for tag, _, position in dep:
            if tag == 'ROOT':
                predicate = self.commit_predicate(tokens[position-1], params)
                break
        return predicate, params

    
    def dep(self, sent):
        return self.model.dependency_parse(sent)


import nltk


class Tokenizer():
    def __init__(self) -> None:
        pass

    def tokenize(self, inp):
        raise NotImplementedError()


class BaseTokenizer(Tokenizer):

    def __init__(self) -> None:
        pass

    def tokenize_(self, inp):
        '''
        :param inp(str | list(str)): input sentence
        :return: tokenized sentence, list(str) if input type is str, 
        list(list(str)) if input type is list(str) 
        '''
        one_line = False
        if isinstance(inp, str):
            inp = [inp]
            one_line = True

        assert isinstance(inp, list)
        assert len(inp) == 0 or isinstance(inp[0], str)

        oup = [i.split(' ') for i in inp]
        if one_line:
            oup = oup[0]
        return oup

    def tokenize(self, inp):
        '''
        :param inp(str | list(str)): input sentence
        :return: tokenized sentence, list(str) if input type is str, 
        list(list(str)) if input type is list(str) 
        '''
        punct = ['.']
        
        if isinstance(inp, str):
            for p in punct:
                inp = inp.replace(p, '')
            oup = inp.split(' ')
            if oup[-1] == '':
                oup = oup[:-1]
            return oup

        if isinstance(inp, list):
            assert isinstance(inp[0], list) or isinstance(inp[0], str)
            return [self.tokenize(i) for i in inp]

        raise ValueError(f'input should be list or str, got {type(inp)}.')
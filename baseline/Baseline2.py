

from .Baseline import Baseline


class BaselineTM(Baseline):

    def __init__(self, task, args_type) -> None:
        super().__init__(task, args_type)
        self.__model = None
    

    def parse_sentence(self, sent):
        params = self.extract_param(sent)
        pass

    def __call__(self):
        return super().__call__()
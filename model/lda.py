import torch
from model.dataset.data_reader import StateData
from model.models import StateModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from functools import reduce
import pickle as pkl
import nltk
import os


class Lda(object):
    def __init__(self, num_topics, dataset: StateData) -> None:
        self.dataset = dataset
        corpus = [nltk.word_tokenize(s) for s in dataset.sents]
        self.dictionary = dataset.dictionary
        for k, v in self.dictionary.token2id.items():
            self.dictionary.id2token[v] = k
        bows = [self.dictionary.doc2bow(s) for s in corpus]
        self.lda = LdaModel(
            corpus=bows, 
            id2word=self.dictionary.id2token,
            eval_every=None,
            num_topics=num_topics
        )

    def predict(self, xs, params, param_candidates, topk=1):
        xs = xs.tolist() if isinstance(xs, torch.Tensor) else xs
        docs = [self.dataset.decode_sentence(s) for s in xs]
        docs = [nltk.word_tokenize(s) for s in docs]
        bows = [self.dictionary.doc2bow(s) for s in docs]

        topics = [sorted(self.lda.get_document_topics(i), key=lambda x: x[1])[-1][:1] for i in bows]
        predicates = [self.lda.get_topic_terms(i[0])[0][:1] for i in topics]
        if '' in [self.dataset.decode_sentence(i) for i in predicates]:
            raise ValueError('\'\' in predicates.')
        params = [i[0] for i in param_candidates]
        return torch.tensor(predicates), params

    def eval_dataset(self):
        gold = reduce(lambda x, y: x+y, [[self.dataset.labels.index(i) for i in j[2]['gold_predicates']] for j in self.dataset])
        ds = reduce(lambda x, y: x+y, [[self.dataset.decode_sentence(i) for i in j[0]] for j in self.dataset])
        bows = [self.dictionary.doc2bow(nltk.word_tokenize(s)) for s in ds]
        infer = [sorted(self.lda.get_document_topics(i), key=lambda x: x[1])[-1][0] for i in bows]
        rand = rand_score(gold, infer)
        adjusted_rand = adjusted_rand_score(gold, infer)

        return {
            'rand_score': rand,
            'adjusted_rand_score': adjusted_rand
        }

    def eval(self):
        '''
        do nothing.
        '''
        return
    
    def save(self, path):
        dir, _ = os.path.split(path)
        os.makedirs(dir, exist_ok=True)
        with open(path, 'wb') as f:
            pkl.dump(self, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = pkl.load(f)
        assert isinstance(obj, cls)
        return obj

    
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParamSorter(nn.Module):

    def __init__(self, feature_in):
        super().__init__()
        self.feature_in = feature_in
        self.rnn = nn.GRU(feature_in, feature_in, batch_first=True)
        self.fc = nn.Linear(feature_in, 2)


    def forward(self, h, param):
        output, hn = self.rnn(param, h.unsqueeze(0))
        prediction = self.fc(output)
        return prediction
        
    def loss(self, pred, gold):
        '''
        gold: param position
        '''
        if len(pred.shape) == 3:
            pred = pred.view(-1, 2)
        if len(gold.shape) == 2:
            gold = gold.view(-1)
        return F.cross_entropy(pred, gold)
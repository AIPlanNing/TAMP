import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence
import math
from data import GernerateDataset
from utils import get_pack

import logging, argparse



'''
给定文本描述的初始状态和目标状态，根据学习到的动作模型，生成动作序列

'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

'''
encode an proposition which is in the formation of (P a a)
'''
class PropEnc(nn.Module):
    def __init__(self, nPredicate, nObject, nAction, nDim):
        super().__init__()
        self.nPredicate = nPredicate
        self.nObject = nObject # one for empty object
        self.nAction = nAction
        self.nDim = nDim
        #self.predicateEmbd = nn.Embedding(nPredicate, nDim)
        #self.objectEmbd = nn.Embedding(self.nObject, self.nDim)
        self.embd = nn.Embedding(self.nObject + self.nPredicate + self.nAction, self.nDim, padding_idx=0)
        self.linear = nn.Sequential(nn.Linear(3*nDim, 2*nDim), nn.ReLU(), nn.Linear(2*nDim, nDim))
        self.rnnEncoder = nn.GRU(nDim, nDim, batch_first=True)
        self.actLinear = nn.Sequential(nn.Linear(nDim*4, nDim*2), nn.ReLU())
    def action_forward(self, action):
        n, l, dim = action.shape
        embd = self.embd(action)
        embd = torch.reshape(embd, (-1, 4, self.nDim))
        hidden = torch.zeros(embd.shape[0], self.nDim).unsqueeze(0).cuda()
        output, hidden = self.rnnEncoder(embd, hidden)
        hidden = torch.reshape(hidden, (n, l, self.nDim))
        embd = torch.reshape(embd, (n, l, -1))
        #return self.actLinear(embd)
        return hidden
    def forward(self, batch):
        '''
        :propMask: to eliminate effect of padding
        '''
        # batch (N L S 3)
        n, l, s = batch.shape[0], batch.shape[1], batch.shape[2]
        embd = self.embd(batch) # (N , L, S, 3, dim)
        embd = torch.reshape(embd, (-1, 3, self.nDim))
        hidden = torch.zeros(embd.shape[0], self.nDim).unsqueeze(0).cuda()
        output, hidden = self.rnnEncoder(embd, hidden)
        hidden = torch.reshape(hidden, (n, l, s, self.nDim))
        embd = torch.reshape(embd, (n, l, s, 3 * self.nDim))# (N, L S 3dim)
        embd = self.linear(embd) # N L S nDim
        return hidden

'''
encode a state which includes several propositions
'''
class StateEnc(nn.Module):
    def __init__(self, nPredicate, nObject, nAction, nDim, nHead, nTrfLayer, dropout):
        super().__init__()
        self.nPredicate = nPredicate
        self.nObject = nObject
        self.nDim = nDim
        self.nHead = nHead
        self.nTrfLayer = nTrfLayer

        self.propEnc = PropEnc(nPredicate, nObject, nAction, nDim)
        self.dense = nn.Sequential(nn.Linear(3 * nDim, nDim), nn.ReLU())
        self.trfEncLayer = nn.TransformerEncoderLayer(nDim, nHead, nDim, dropout=dropout)
        self.trsfEnc = nn.TransformerEncoder(self.trfEncLayer, nTrfLayer)
    def forward(self, batch: torch.Tensor, propMask: torch.Tensor):
        # batch N L S 3

        bs, length, stateLen, dim = batch.shape
        propRep = self.propEnc(batch) # n, l, s, dim
        dim = propRep.shape[-1]
        stack = propRep.reshape((bs, -1, dim)) # n, l*s, dim
        stack = stack.permute((1, 0, 2)) #l*s, n, dim

        trfMask = propMask.reshape((bs, -1)) # n, L*s
        trfMask = trfMask.permute((1, 0)) # l*s, n

        out: torch.Tensor = self.trsfEnc(stack) #l*s, n, dim
        out = out.permute((1, 0, 2)) # N, L*S, dim
        out = out.reshape((bs, length, stateLen, -1)) # n, l, s, dim

        out[propMask > 0] = 0
        stateLen = (propMask == 0).sum(dim=2, keepdim=True) # n, l
        #mean = out.sum(dim=2) / stateLen # N, L, dim
        mean = out.sum(dim=2)

        seqLen = ((propMask == 0).sum(2) != 0).sum(1)
        #pack = pack_padded_sequence(mean, seqLen, batch_first=True, enforce_sorted=True)
        #assert(out.shape[0] == bs and out.shape[1] == dim and len(out.shape) == 2)
        return mean

class ActionGenerator(nn.Module):
    def __init__(self, nPredicate, nObject, nAction, nDim, nHead, nTrfLayer, dropout):
        super().__init__()
        self.nPredicate, self.nObject, self.nAction, self.nDim, self.nHead, self.nTrfLayer = nPredicate, nObject, nAction, nDim, nHead, nTrfLayer
        self.nSymbol = self.nPredicate + self.nAction + self.nObject

        self.stateEnc = StateEnc(nPredicate, nObject, nAction, nDim, nHead, nTrfLayer, dropout=dropout)

        self.rnn = nn.GRU(input_size=nDim, hidden_size=nDim, num_layers=1, batch_first=True)
        self.zip = nn.Linear(nDim*2, nDim)
        # self.linear = nn.Sequential(nn.Linear(2*nDim, nDim), nn.ReLU())
        
        self.actionHeuristic = nn.Linear(nDim, self.nSymbol + 1)
        self.object1Heuristic = nn.Linear(nDim, self.nSymbol + 1)
        self.object2Heuristic = nn.Linear(nDim, self.nSymbol + 1)
        self.object3Heuristic = nn.Linear(nDim, self.nSymbol + 1)
        

    def forward(self, batch, propMask, actions):
        # batch N L S 3 # L: action sequenth length, S: proposition length in a state
        out = self.stateEnc(batch, propMask) # n, l, dim
        seqLen = ((propMask == 0).sum(2) != 0).sum(1)
        goal = out[torch.arange(0, batch.shape[0]), seqLen-1, :]
        init = out[:, 0, :]
        '''outLeftSide = out[:, :-1, :]    # n, l-1, dim (sequence index 0 to len-2)
        outRightSide = out[:, 1:, :]    # n, l-1, dim (sequence index 1 to len-1)
        cat = torch.cat((outLeftSide, outRightSide), dim=2) # n, l-1, 2*dim'''
        
        actionsEnc = self.stateEnc.propEnc.action_forward(actions)
        packedSeq = get_pack(actionsEnc, seqLen)
        h0 = torch.cat((init, goal), dim=1).unsqueeze_(0)
        h0 = self.zip(h0)

        output, hidden = self.rnn(packedSeq, h0)


        action = self.actionHeuristic(output.data)
        object1 = self.object1Heuristic(output.data)
        object2 = self.object2Heuristic(output.data)
        object3 = self.object3Heuristic(output.data)
        return action, object1, object2, object3

    def inference(self, batch, propMask, actions):
        out = self.stateEnc(batch, propMask)
        seqLen = ((propMask == 0).sum(2) != 0).sum(1)
        goal = out[torch.arange(0, batch.shape[0]), seqLen-1, :]
        init = out[:, 0, :]
        h0 = torch.cat((init, goal), dim=1).unsqueeze_(0)
        h0 = self.zip(h0)
        
        actionsEnc = self.stateEnc.propEnc.action_forward(actions)[:, [0], :]
        output  = actionsEnc
        ret = []

        for i in range(20):
            output, h0 = self.rnn(output, h0)
            
            action = self.actionHeuristic(output).topk(1)[1].squeeze()
            obj1 = self.object1Heuristic(output).topk(1)[1].squeeze()
            obj2 = self.object2Heuristic(output).topk(1)[1].squeeze()
            obj3 = self.object3Heuristic(output).topk(1)[1].squeeze()
            
            ret.append(torch.stack((action, obj1, obj2, obj3), 1))
        return torch.stack(ret).permute((1, 0, 2))



if __name__ == "__main__":
    #dataset = GernerateDataset("nb", "train")
    #loader = DataLoader(dataset=dataset, batch_size=args.batch_size)
    model = ActionGenerator(4, 4, 4, 128, 8, 6, 0.2)
    inp = torch.randint(0, 12, (32, 10, 12, 3))
    mask = torch.randint(0, 12, (32, 10, 12))
    oup = model(inp, mask, 0)
    print("done")
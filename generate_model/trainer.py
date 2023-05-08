from data import GernerateDataset
from model import ActionGenerator
from utils import get_pack
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence, PackedSequence
import math
import logging, argparse
import tqdm
import os

import time, datetime

def padToLength(inp, length, padid=0):
    def getPad(template):
        if isinstance(template, int):
            return padid
        if isinstance(template, list):
            return [getPad(template[0]) for i in template]


    pad = getPad(inp[0])
    ret = inp + [pad]*length
    return ret[:length]
    

def collectData(data):
    #propositions = [[trainDataset.symbolEncode(i) for i in plan["gold_proposition"]] for plan in data]
    propositions = [padToLength([padToLength([trainDataset.symbolEncode(pred) for pred in state], args.state_len) for state in plan["gold_proposition"]], args.seq_len, 0) for plan in data]
    propositions = torch.tensor(propositions).long()
    propMask = (propositions == 0).sum(dim=3)
    lenProp = [[len(state) for state in plan["gold_proposition"]] for plan in data]
    actions = [[trainDataset.symbolEncode(action, 4) for action in [trainDataset.symbolTokenizer.boa] + plan["action"] + [trainDataset.symbolTokenizer.eoa]] for plan in data]

    lenAct = [len(plan["action"]) for plan in data]
    return propositions, actions

def makeBatch(action: list, return_pack=False):
    def get_label(action, index=0):
        label = list(map(lambda x: torch.tensor([i[index] for i in x[1:]]), action))
        length = torch.tensor([len(i) for i in label]).long()
        pad = pad_sequence(label, True).unsqueeze(2)
        ret = get_pack(pad, length)
        if return_pack:
            return ret
        else:
            return ret.data.squeeze()

    input = [torch.tensor(sequence[:-1]).long() for sequence in action]

    sequenceLen = [len(sequence) for sequence in action]

    actLabel = get_label(action, 0)
    obj1label = get_label(action, 1)
    obj2label = get_label(action, 2)
    obj3label = get_label(action, 3)


    label = {"act": actLabel, "obj1": obj1label, "obj2": obj2label, "obj3": obj3label}
    input = pad_sequence(input, batch_first=True)
    return input, label, sequenceLen

def inference(model: nn.Module, loader, device, symbolTokenizer):
    model.eval()  # turn on evaluation mode
    with torch.no_grad():
        for propositions, actions in loader:
            propMask = (propositions == 0).sum(dim=3)
            actInp, labels, actLen = makeBatch(actions, return_pack=True)
            propositions = propositions.to(device)
            propMask = propMask.to(device)
            actInp = actInp.to(device)
            actLabelPack = labels["act"]
            obj1LabelPack = labels["obj1"]
            obj2LabelPack = labels["obj2"]
            obj3LabelPack = labels["obj3"]
            ret = model.inference(propositions, propMask, actInp)
            
            act, obj1, obj2, obj3 = labels["act"], labels["obj1"], labels["obj2"], labels["obj3"]
            #actOupPack = PackedSequence(data=actOup, batch_sizes=act.batch_sizes)
            #obj1OupPack = PackedSequence(data=obj1Oup, batch_sizes=obj1.batch_sizes)
            #obj2OupPack = PackedSequence(data=obj2Oup, batch_sizes=obj2.batch_sizes)
            #obj3OupPack = PackedSequence(data=obj3Oup, batch_sizes=obj3.batch_sizes)
            '''actLabelPack = PackedSequence(data=actLabel, batch_sizes=labels["act"].batch_sizes)
            obj1LabelPack = PackedSequence(data=obj1Label, batch_sizes=labels["obj1"].batch_sizes)
            obj2LabelPack = PackedSequence(data=obj2Label, batch_sizes=labels["obj2"].batch_sizes)
            obj3LabelPack = PackedSequence(data=obj3Label, batch_sizes=labels["obj3"].batch_sizes)
'''
            

            #actPad = pad_packed_sequence(actOupPack, True)[0]
            #obj1Pad = pad_packed_sequence(obj1OupPack, True)[0]
            #obj2Pad = pad_packed_sequence(obj2OupPack, True)[0]
            #obj3Pad = pad_packed_sequence(obj3OupPack, True)[0]
            actLabelPad = pad_packed_sequence(actLabelPack, True)[0].to(device)
            obj1LabelPad = pad_packed_sequence(obj1LabelPack, True)[0].to(device)
            obj2LabelPad = pad_packed_sequence(obj2LabelPack, True)[0].to(device)
            obj3LabelPad = pad_packed_sequence(obj3LabelPack, True)[0].to(device)

            seqLen = ((propMask == 0).sum(2) != 0).sum(1)
            sorted_len, sorted_idx = seqLen.sort(dim=0, descending=True)
            _, original_idx = sorted_idx.sort(dim=0, descending=False)
            unsorted_idx = original_idx.view(-1, 1, 1).expand_as(actLabelPad)
            #actPad = torch.gather(actPad, index=unsorted_idx, dim=0).contiguous()
            #obj1Pad = torch.gather(obj1Pad, index=unsorted_idx, dim=0).contiguous()
            #obj2Pad = torch.gather(obj2Pad, index=unsorted_idx, dim=0).contiguous()
            #obj3Pad = torch.gather(obj3Pad, index=unsorted_idx, dim=0).contiguous()
            actLabel = torch.gather(actLabelPad, index=unsorted_idx, dim=0).contiguous()
            obj1Label = torch.gather(obj1LabelPad, index=unsorted_idx, dim=0).contiguous()
            obj2Label = torch.gather(obj2LabelPad, index=unsorted_idx, dim=0).contiguous()
            obj3Label = torch.gather(obj3LabelPad, index=unsorted_idx, dim=0).contiguous()

            #batchOup = torch.cat((actPad.topk(1, 2)[1], obj1Pad.topk(1, 2)[1], obj2Pad.topk(1, 2)[1], obj3Pad.topk(1, 2)[1]), 2)
            batchLabel = torch.cat((actLabel, obj1Label, obj2Label, obj3Label), 2)

            #oup = symbolTokenizer.decode(batchOup)
            label = symbolTokenizer.decode(batchLabel)
    return 

def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for propositions, actions in loader:
            propMask = (propositions == 0).sum(dim=3)
            actInp, labels, actLen = makeBatch(actions)
            propositions = propositions.to(device)
            propMask = propMask.to(device)
            actInp = actInp.to(device)
            actLabel = labels["act"].to(device)
            obj1Label = labels["obj1"].to(device)
            obj2Label = labels["obj2"].to(device)
            obj3Label = labels["obj3"].to(device)
            actOup, obj1Oup, obj2Oup, obj3Oup = model(propositions, propMask, actInp)

            actLoss = criterion(actOup, actLabel)
            obj1Loss = criterion(obj1Oup, obj1Label)
            obj2Loss = criterion(obj2Oup, obj2Label)
            obj3Loss = criterion(obj3Oup, obj3Label)
            loss = (actLoss + obj1Loss + obj2Loss + obj3Loss) / 4
            
            total_loss += args.bs * loss.item()
    return total_loss / (len(loader) - 1)

def train(model, loader, device, optimizer, scheduler, criterion, epoch):
    model.train() # Turn on the train mode
    model.to(device)
    total_loss = 0.
    start_time = time.time()
    batch = 0
    for propositions, actions in loader:
        propMask = (propositions == 0).sum(dim=3)
        actInp, labels, actLen = makeBatch(actions)
        propositions = propositions.to(device) 
        propMask = propMask.to(device)
        actInp = actInp.to(device)
        actLabel = labels["act"].to(device)
        obj1Label = labels["obj1"].to(device)
        obj2Label = labels["obj2"].to(device)
        obj3Label = labels["obj3"].to(device)


        optimizer.zero_grad()


        actOup, obj1Oup, obj2Oup, obj3Oup = model(propositions, propMask, actInp)
        actLoss = criterion(actOup, actLabel)
        obj1Loss = criterion(obj1Oup, obj1Label)
        obj2Loss = criterion(obj2Oup, obj2Label)
        obj3Loss = criterion(obj3Oup, obj3Label)
        loss = (actLoss + obj1Loss + obj2Loss + obj3Loss) / 4
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 20
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl '.format(
                    epoch, batch, len(train_loader), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss))
            total_loss = 0
            start_time = time.time()

        batch += 1

def check_args(args):
    os.makedirs(args.model_dir, exist_ok=True)

def save(model, optimizer, scheduler, epoch, loss, args):
    now = datetime.datetime.now()
    os.makedirs(args.model_dir, exist_ok=True)
    path = os.path.join(args.model_dir, f"checkpoint-{args.task}-{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss
    }, path)
    torch.save(args, os.path.join(args.model_dir, "args"))

def load(path, model, optimizer, scheduler):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    loss = ckpt["loss"]
    return model, optimizer, scheduler, epoch, loss
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--task", type=str, default="nb")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32, help="max length of a plan trace")
    parser.add_argument("--state_len", type=int, default=20, help="max length of a state")

    # model
    parser.add_argument("--nDim", type=int, default=128, help="dimension of hidden state")
    parser.add_argument("--nHead", type=int, default=4, help="number of head")
    parser.add_argument("--nTrfLayer", type=int, default=1, help="num of transformer layers")
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--model_dir", default="tmp/action_generate_model", type=str)
    parser.add_argument("--from_file", default="", type=str)

    # train
    parser.add_argument("--epoch", default=64)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.95)
    args = parser.parse_args()
    trainDataset = GernerateDataset(args.task, "train")
    devDataset = GernerateDataset(args.task, "dev")
    testDataset = GernerateDataset(args.task, "test")

    train_loader = DataLoader(trainDataset, args.bs, shuffle=True, collate_fn=collectData)
    dev_loader = DataLoader(devDataset, args.bs, shuffle=True, collate_fn=collectData)

    criterion = nn.CrossEntropyLoss()
    taskinfo = trainDataset.taskInfo
    model = ActionGenerator(len(taskinfo.predicates), len(taskinfo.objects) + 1, len(taskinfo.actions) + 2, args.nDim, args.nHead, args.nTrfLayer, args.dropout)
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=args.gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    startEpoch = 0
    best_val_loss = float('inf')
    if args.from_file:
        path = os.path.join(args.model_dir, args.from_file)
        model, optimizer, scheduler, startEpoch, loss = load(path, model, optimizer, scheduler)
    for epoch in range(startEpoch, args.epoch):
        epoch_start_time = time.time()
        train(model, train_loader, device, optimizer, scheduler, criterion, epoch)

        val_loss = evaluate(model, dev_loader, device)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} |')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save(model, optimizer, scheduler, epoch, val_loss, args)
        scheduler.step()
    pass
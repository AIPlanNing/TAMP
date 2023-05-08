from trainer import inference, padToLength
from model import ActionGenerator
from data import GernerateDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import argparse

def collectData(data):
    #propositions = [[trainDataset.symbolEncode(i) for i in plan["gold_proposition"]] for plan in data]
    propositions = [padToLength([padToLength([trainDataset.symbolEncode(pred) for pred in state], args.state_len) for state in plan["gold_proposition"]], args.seq_len, 0) for plan in data]
    propositions = torch.tensor(propositions).long()
    propMask = (propositions == 0).sum(dim=3)
    lenProp = [[len(state) for state in plan["gold_proposition"]] for plan in data]
    actions = [[trainDataset.symbolEncode(action, 4) for action in [trainDataset.symbolTokenizer.boa] + plan["action"] + [trainDataset.symbolTokenizer.eoa]] for plan in data]

    lenAct = [len(plan["action"]) for plan in data]
    return propositions, actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="tmp/action_generate_model/checkpoint-nb-0.pt")
    # data
    parser.add_argument("--task", type=str, default="nb")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32, help="max length of a plan trace")
    parser.add_argument("--state_len", type=int, default=20, help="max length of a state")

    # model
    parser.add_argument("--nDim", type=int, default=128, help="dimension of hidden state")
    parser.add_argument("--nHead", type=int, default=4, help="number of head")
    parser.add_argument("--nTrfLayer", type=int, default=1, help="num of transformer layers")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--model_dir", default="tmp/action_generate_model", type=str)
    parser.add_argument("--from_file", default="", type=str)

    # train
    parser.add_argument("--epoch", default=64)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95)
    
    args = parser.parse_args()
    trainDataset = GernerateDataset(args.task, "train")
    devDataset = GernerateDataset(args.task, "dev")
    testDataset = GernerateDataset(args.task, "test")

    train_loader = DataLoader(trainDataset, args.bs, shuffle=True, collate_fn=collectData)
    dev_loader = DataLoader(devDataset, args.bs, shuffle=True, collate_fn=collectData)
    test_loader = DataLoader(testDataset, args.bs, shuffle=False, collate_fn=collectData)
    taskinfo = trainDataset.taskInfo
    model = ActionGenerator(len(taskinfo.predicates), len(taskinfo.objects) + 1, len(taskinfo.actions) + 2, args.nDim, args.nHead, args.nTrfLayer, args.dropout)
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inference(model, test_loader, device, trainDataset.symbolTokenizer)

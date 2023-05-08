from ..plan_generator import ActionModel

import torch
import torch.nn

import argparse, logging
logger = logging.getLogger()
parser = argparse.ArgumentParser()
args = parser.parse_args()



if __name__ == "__main__":
    action_model = args.am_file # input action model file in pddl format
    with open(args.am_file, "r") as f:
        action_model = ActionModel.parse(f.read())
    
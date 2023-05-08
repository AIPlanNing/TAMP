from torch.nn.utils.rnn import pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F




def get_mask_for_transformer(**kwargs):
    pack = kwargs.get('pack')
    if not pack:
        pad = kwargs.get('pad')
        lengths = kwargs.get('lengths')
        if not pad or not lengths:
            raise ValueError(f'except input\n pack(PackedSequence)\n or \n  pad(torch.Tensor) lengths(list)')
    else:
        pad, lengths = pad_packed_sequence(pack, batch_first=True)
    mask = torch.ones((pad.shape[0], pad.shape[1])).to(torch.bool)
    for row_idx, length in enumerate(lengths):
        for column_idx in range(length):
            mask[row_idx][column_idx] = False
    return pad, mask
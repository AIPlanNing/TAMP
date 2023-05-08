from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
import torch

def get_pack(input, seqLen):
    sorted_len, sorted_idx = seqLen.sort(dim=0, descending=True)
    index_sorted_idx = sorted_idx.view(-1, 1,1).expand_as(input)
    sorted_inputs = torch.gather(input, index=index_sorted_idx.long(), dim=0)
    packed_seq = pack_padded_sequence(sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)
    return packed_seq

def flatten_sequence(input, seqLen):
    pass
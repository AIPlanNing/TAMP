import types
import nltk
from model.dataset.data_reader import TextActionModelData

def format_sent_with_args(sent: str, args_type_set: dict, mask_prefix: str='arg') -> str:
    # tokens = sent.split(' ')
    tokens = nltk.word_tokenize(sent)
    arg_count = 0
    for token_id, token in enumerate(tokens):
        for arg, t in args_type_set.items():
            if arg == token:
                tokens[token_id] = f'<{t}_arg{arg_count}>'
                arg_count += 1
    return ' '.join(tokens)



if __name__ == '__main__':
    f = format_sent_with_args('block1 is on block2.', args_type_set={'block1': 1, 'block2': 1, 'block3': 1, 'robot': 0})
    print(f)
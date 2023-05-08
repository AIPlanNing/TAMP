def check_config(args):
    if not args.embedding_from_wv and args.embedding_freeze:
        raise ValueError('Freeze random initialize embedding layer is not allow.\
            Set args.embedding_from_wv True or args.embedding_freeze False.\
            ')
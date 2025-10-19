from .random import RandomDP


def get_data_pruner(principle, dataset, model, tokenizer):

    if principle == 'random':
        return RandomDP(dataset)
    else:
        raise ValueError('NOT SUPPORTED PRINCIPLE')

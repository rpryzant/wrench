
def get_params(dataset, expt_type='best'):

    if dataset == 'agnews':
        return 16, 1.2, 80, 50, 'ngram', True, False, 0.8

    elif dataset == 'cdr':
        return 16, 1, 80, 0, 'ngram-pca', True, False, 0.8

    elif dataset == 'chemprot':
        return 16, 1, 80, 50, 'ngram-tree', True, True, 0.8

    elif dataset == 'imdb':
        return 32, 1, 80, 50, 'ngram-tree', True, False, 0.8

    elif dataset == 'scicite':
        return 32, 1.2, 80, 0, 'ngram-pca', True, False, 0.95

    elif dataset == 'semeval':
        return 32, 1, 80, 50, 'ngram', False, True, 0.8

    elif dataset == 'sms':
        return 16, 1.2, 80, 50, 'ngram-pca', False, True, 0.8

    elif dataset == 'trec':
        return 32, 1.2, 80, 50, 'ngram-pca', False, False, 0.8

    elif dataset == 'youtube':
        return 16, 1, 80, 50, 'ngram', True, False, 0.8

    


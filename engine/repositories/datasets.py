from engine.training_engine.dataset_loaders.cora import load_cora

DATASETS = {
    'cora': {
        'task': 'node',
        'metric': 'Accuracy',
        'loader': load_cora
    },
    'pubmed': {
        'task': 'node',
        'metric': 'Accuracy'
    },
    'ogbn-arxiv': {
        'task': 'node',
        'metric': 'Accuracy'
    },
    'ogbn-products': {
        'task': 'node',
        'metric': 'Accuracy'
    },
    'ogbn-papers100M': {
        'task': 'node',
        'metric': 'Accuracy'
    },
    'mutag': {
        'task': 'graph',
        'metric': 'Accuracy'
    },
    'collab': {
        'task': 'graph',
        'metric': 'Accuracy'
    },
    'ogbg-molhiv': {
        'task': 'graph',
        'metric': 'ROC-AUC'
    },
    'ogbg-molpcba': {
        'task': 'graph',
        'metric': 'AP'
    },
    'ogbg-code2': {
        'task': 'graph',
        'metric': 'F1 score'
    },
    'ogbl-ddi': {
        'task': 'link',
        'metric': 'Hits@20'
    },
    'ogbl-collab': {
        'task': 'link',
        'metric': 'Hits@50'
    },
    'ogbl-ppa': {
        'task': 'link',
        'metric': 'Hits@100'
    },
    'ogbl-wikikg2': {
        'task': 'link',
        'metric': 'MRR'
    },
    'ogbl-vessel': {
        'task': 'link',
        'metric': 'ROC-AUC'
    }
}

DATASETS_LIST = [*DATASETS]

NODE_CLASS_DATASETS = [model for model, data in DATASETS.items() if 'node' in data['task']]
LINK_CLASS_DATASETS = [model for model, data in DATASETS.items() if 'link' in data['task']]
GRAPH_CLASS_DATASETS = [model for model, data in DATASETS.items() if 'graph' in data['task']]

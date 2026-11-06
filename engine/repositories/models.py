from engine.paths import MODELS_CONFIG_DIR


def build_gcn(**kwargs):
    """Load the model implementation only when an experiment needs it."""
    from engine.training_engine.models.gcn import GCN

    return GCN(**kwargs)

MODELS = {
    'gcn': {
        'object': build_gcn,
        'task': ['node'],
        'config_dir': MODELS_CONFIG_DIR / "gcn"
    },
    'gat': {
        'task': ['node'],
        'config_dir': MODELS_CONFIG_DIR / "gat"
    },
    'sage': {
        'task': ['node'],
        'config_dir': MODELS_CONFIG_DIR / "sage"
    },
    'gin': {
        'task': ['graph'],
        'config_dir': MODELS_CONFIG_DIR / "gin"
    },
    'mincutpool': { 
        'task': ['graph'],
        'config_dir': MODELS_CONFIG_DIR / "mincutpool"
    },
    'structpool': {
        'task': ['graph'],
        'config_dir': MODELS_CONFIG_DIR / "structpool"
    },
    'empool': {
        'task': ['node', 'graph'],
        'config_dir': MODELS_CONFIG_DIR / "empool"
    }
}

MODELS_LIST = [*MODELS]

NODE_CLASS_MODELS = [model for model, metadata in MODELS.items() if 'node' in metadata['task']]
LINK_CLASS_MODELS = [model for model, metadata in MODELS.items() if 'link' in metadata['task']]
GRAPH_CLASS_MODELS = [model for model, metadata in MODELS.items() if 'graph' in metadata['task']]

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'unfreeze_from': {
            'value': 'layer4'
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_decay': {
            'values': [0, 1e-5, 1e-4]
        },
        'optimizer': {
            'values': ['adam', 'nadam', 'rmsprop', 'sgd']
        },
        'epochs': {
            'value': 10
        }
    }
}

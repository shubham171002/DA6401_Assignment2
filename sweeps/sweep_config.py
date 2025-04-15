sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },
    "parameters": {
        "num_filters": { "values": [16, 32, 64] },
        "activation_fn": { "values": ["relu", "gelu", "silu", "mish"] },
        "dense_neurons": { "values": [64, 128, 256] },
        "dropout": { "values": [0.1, 0.2, 0.3] },
        "batchnorm": { "values": [True, False] },
        "filter_organization": { "values": ["same", "double", "half"] },
        "learning_rate": { "max": 1e-3, "min": 1e-4 },
        "data_aug": { "values": [True, False] }
    }
}

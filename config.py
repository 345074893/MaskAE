class Config:
    def __init__(self):
        self.hidden_size = 700
        self.generateor_learning_rate = 5e-4
        self.discriminator_lerning_rate = 5e-4
        self.learning_rate = 5e-4
        self.lambda_g=0.1
        self.gamma_decay=0.5

        
        self.model = {
            "dim_c": 200,
            "dim_z": 500,
            "embed_size": 100,
            "rnn_lm": {
            "rnn_cell": {
                "type": "GRUCell",
                "num_units": 700,
                "dropout": 0.5
            }
            },
            "classifier": {
            "kernel_size": [3, 4, 5],
            "filters": 128,
            "other_conv_kwargs": {"padding": "same"},
            "dropout_conv": [1],
            "dropout_rate": 0.5,
            "num_dense_layers": 0,
            "num_classes": 1
            },
            "opt": {
            "optimizer": {
                "type":  "AdamOptimizer",
                "kwargs": {
                    "learning_rate": 1e-4,
                },
            },
            },
            }
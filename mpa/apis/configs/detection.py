from sc_sdk.sc_sdk.configuration.configurable_parameters import Float, Integer, Selectable
from sc_sdk.sc_sdk.configuration.configurable_parameters import Group, Option, Object

from apis.configs.base import MPABaseParameters


class MPADetectionParameters(MPABaseParameters):
    class __LearningParameters(Group):
        header = "Learning Parameters"
        description = header

        batch_size = Integer(
            default_value=5,
            min_value=1,
            max_value=32,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "reduces training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            editable=True,
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
        )

        num_epochs = Integer(
            default_value=1,
            header="Number of epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            min_value=1,
            max_value=1000,
            editable=True,
        )

        learning_rate = Float(
            default_value=0.0025,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable. A value "
                        "of 0.0025 is recommended.",
            min_value=1e-06,
            max_value=1e-02,
            editable=True,
        )

        learning_rate_warmup_iters = Integer(
            default_value=100,
            header="Number of iterations for learning rate warmup",
            description="",
            min_value=0,
            max_value=1000,
            editable=True,
        )

        learning_rate_schedule = Selectable(header="Learning rate schedule",
                                            default_value="exp",
                                            options=[Option(key="fixed",
                                                            value="Fixed",
                                                            description="Learning rate is kept fixed during training"),
                                                     Option(key="exp",
                                                            value="Exponential annealing",
                                                            description="Learning rate is reduced exponentially"),
                                                     Option(key="step",
                                                            value="Step-wise annealing",
                                                            description="Learning rate is reduced step-wise at "
                                                                        "epoch 10"),
                                                     Option(key="cyclic",
                                                            value="Cyclic cosine annealing",
                                                            description="Learning rate is gradually reduced and "
                                                                        "increased during training, following a cosine "
                                                                        "pattern. The pattern is repeated for two "
                                                                        "cycles in one training run."),
                                                     Option(key="custom",
                                                            value="Custom",
                                                            description="Learning rate schedule that is provided "
                                                                        "with the model."),
                                                     ],
                                            description="Specify learning rate scheduling for the MMDetection task. "
                                                        "When training for a small number of epochs (N < 10), the fixed"
                                                        " schedule is recommended. For training for 10 < N < 25 epochs,"
                                                        " step-wise or exponential annealing might give better results."
                                                        " Finally, for training on large datasets for at least 20 "
                                                        "epochs, cyclic annealing could result in the best model.",
                                            editable=True)

    class __AlgoBackend(Group):
        header = "Internal Algo Backend parameters"
        description = header
        template = Selectable(header="Model template file",
                              default_value='template',
                              options=[Option(key='template', value='template',
                                              description='Path to model template file')],
                              description="Model template.",
                              editable=False)
        model_name = Selectable(header="Model name",
                                default_value='model',
                                options=[Option(key='model', value='model',
                                                description='Model name')],
                                description="Specify model name.",
                                editable=False)
        model = Selectable(header="Model architecture",
                           default_value='model.py',
                           options=[Option(key='model.py', value='model.py',
                                           description='Path to model configuration file')],
                           description="Specify learning architecture for the the task.",
                           editable=False)
        data_pipeline = Selectable(header="Data pipeline",
                                   default_value='ote_data_pipeline.py',
                                   options=[Option(key='ote_data_pipeline.py', value='ote_data_pipeline.py',
                                                   description='Path to data pipeline configuration file')],
                                   description="Specify data pipeline for the the task.",
                                   editable=False)

    learning_parameters: __LearningParameters = Object(__LearningParameters)
    algo_backend: __AlgoBackend = Object(__AlgoBackend)

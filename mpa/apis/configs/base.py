from sc_sdk.configuration.configurable_parameters import Selectable
from sc_sdk.configuration.configurable_parameters import Group, Option, Object
from sc_sdk.configuration.deep_learning_configurable_parameters import DeepLearningConfigurableParameters


class MPABaseParameters(DeepLearningConfigurableParameters):
    class __MPAParameters(Group):
        header = "MPA parameters"
        description = header

        recipe = Selectable(
            header='MPA recipe file',
            default_value='recipe.yaml',
            options=[
                Option(
                    key='recipe',
                    value='recipe',
                    description='path to recipe file')
            ],
            description='MPA recipe file',
            editable=False
        )

    mpa_parameters: __MPAParameters = Object(__MPAParameters)

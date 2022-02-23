from mpa.registry import STAGES
from .stage import DetectionStage


@STAGES.register_module()
class DetectionExporter(DetectionStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def export(self, **kwargs):
        raise NotImplementedError

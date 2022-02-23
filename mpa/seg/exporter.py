from mpa.registry import STAGES
from .stage import SegStage


@STAGES.register_module()
class SegExporter(SegStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def export(self, **kwargs):
        raise NotImplementedError

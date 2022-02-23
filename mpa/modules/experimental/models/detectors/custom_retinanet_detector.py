from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.retinanet import RetinaNet
from mpa.modules.models.detectors.sam_detector_mixin import SAMDetectorMixin
from mpa.modules.models.detectors.l2sp_detector_mixin import L2SPDetectorMixin


@DETECTORS.register_module()
class CustomRetinaNet(SAMDetectorMixin, L2SPDetectorMixin, RetinaNet):
    """SAM optimizer & L2SP regularizer enabled custom RetinaNet
    """
    pass

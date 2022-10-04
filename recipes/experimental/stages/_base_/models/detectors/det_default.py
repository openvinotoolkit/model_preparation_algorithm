_base_ = [
    '../../../../../../samples/cfgs/models/detectors/atss_mv2w1.custom.yaml',
]

load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-atss.pth'
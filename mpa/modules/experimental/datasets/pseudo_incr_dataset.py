from mmdet.datasets import PIPELINES, DATASETS, CocoDataset
from mmdet.datasets.pipelines import to_tensor, MinIoURandomCrop
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmcv.parallel import DataContainer as DC

# import torch
import numpy as np
from numpy import random


@DATASETS.register_module()
class PseudoIncrCocoDataset(CocoDataset):
    """COCO dataset w/ pseudo label augmentation
    """

    def __init__(self, pre_stage_res, pseudo_threshold=0.9, **kwargs):

        # Build org dataset
        dataset_cfg = kwargs.copy()
        _ = dataset_cfg.pop('org_type', None)
        super().__init__(**dataset_cfg)

        # Load pseudo labels
        self.pseudo_file = pre_stage_res
        self.pseudo_threshold = pseudo_threshold
        self.pseudo_data = np.load(self.pseudo_file, allow_pickle=True).item()
        self.pseudo_anns = self.pseudo_data['detections']
        self.pseudo_classes = self.pseudo_data['classes']
        self.CLASSES = list(self.pseudo_classes) + list(self.CLASSES)
        print(f'PseudoLabelIncrDataset!!: {self.CLASSES}')
        self.statistics()

    def get_ann_info(self, idx):
        """Overriding CocoDataset.get_ann_info()
        """

        ann_info = super().get_ann_info(idx)

        # Adjust class labels
        ann_info['labels'] += len(self.pseudo_classes)  # Addin # old classes
        # TODO: This could be class name sensitive adaptation

        # Format pseudo labels
        pseudo_ann = self.pseudo_anns[idx]  # [img][label][bbox]
        labels = []
        bboxes = []
        probs = []
        for label, detections in enumerate(pseudo_ann):
            for detection in detections:
                if detection[4] > self.pseudo_threshold:
                    labels.append(label)
                    bboxes.append(detection[:4])
                    # probabilities for old classes including BG at the end
                    probs.append(detection[5:])

        # print('new_labels', ann_info['labels'])
        # print('old_labels', np.array(labels))
        # print('new_bboxes', ann_info['bboxes'])
        # print('old_bboxes', np.array(bboxes))
        # print(f"old: {len(labels)}, new: {ann_info['labels'].shape[0]}")
        num_org_bboxes = ann_info['labels'].shape[0]
        new_probs = np.zeros((num_org_bboxes, len(self.pseudo_classes) + 1), dtype=np.float32)
        new_probs[:, -1] = 1.0  # NEW classes are BG for OLD classes
        if len(labels) > 0:
            ann_info['labels'] = \
                np.concatenate((ann_info['labels'], np.array(labels)))
            ann_info['bboxes'] = \
                np.concatenate((ann_info['bboxes'], np.array(bboxes)))
            ann_info['pseudo_labels'] = \
                np.concatenate((new_probs, np.array(probs)))
        else:
            ann_info['pseudo_labels'] = new_probs

        return ann_info

    def statistics(self):
        num_bboxes = 0
        num_labels = np.zeros(len(self.pseudo_classes))
        prob_acc = np.zeros(len(self.pseudo_classes) + 1)
        for pseudo_ann in self.pseudo_anns:
            for label, detections in enumerate(pseudo_ann):
                for detection in detections:
                    if detection[4] > self.pseudo_threshold:
                        num_bboxes += 1
                        num_labels[label] += 1
                        prob_acc += detection[5:]
        print('pseudo label stat')
        print(f'- # images: {len(self.pseudo_anns)}')
        print(f'- # bboxes: {num_bboxes}')
        print(f'- # labels: {num_labels}')
        if num_bboxes > 0:
            print(f'- label ratio: {num_labels / num_bboxes}')
            print(f'- prob ratio: {prob_acc / float(num_bboxes)}')


@PIPELINES.register_module()
class FormatPseudoLabels(object):
    """Data processor for pseudo label formatting
    """

    def __init__(self):
        print('Init FormatPseudoLabels')

    def __call__(self, data):
        plabels = data['pseudo_labels']
        # tensor -> DataContainer
        data['pseudo_labels'] = DC(to_tensor(plabels))
        return data


@PIPELINES.register_module()
class LoadPseudoLabels(object):
    """Data processor for pseudo label formatting
    """

    def __init__(self):
        print('Init FormatPseudoLabels')

    def __call__(self, data):
        plabels = data['ann_info']['pseudo_labels']
        # tensor -> DataContainer
        data['pseudo_labels'] = plabels
        return data


@PIPELINES.register_module()
class PseudoMinIoURandomCrop(MinIoURandomCrop):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        super(PseudoMinIoURandomCrop, self).__init__(min_ious, min_crop_size, bbox_clip_border)
        self.bbox2label = {
            'gt_bboxes': ('gt_labels', 'pseudo_labels'),
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': ('gt_labels', 'pseudo_labels'),
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        img = results['img']
        if 'bbox_fields' not in results:
            raise KeyError('bbox_fields is not in results')
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        if self.bbox_clip_border:
                            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_keys = self.bbox2label.get(key)
                        if isinstance(label_keys, tuple):
                            for label_key in label_keys:
                                if label_key in results:
                                    results[label_key] = results[label_key][mask]
                        else:
                            if label_keys in results:
                                results[label_keys] = results[label_keys][mask]

                        # mask fields
                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3], patch[0]:patch[2]]
                return results

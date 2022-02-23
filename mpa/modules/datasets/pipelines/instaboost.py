import numpy as np
import instaboostfast as instaboost
from mmdet.datasets.builder import PIPELINES

import mmcv
import pycocotools.mask as cocomask
from skimage import measure


@PIPELINES.register_module()
class InstaBoostMPA(object):
    """
    Data augmentation method in paper "InstaBoost: Boosting Instance
    Segmentation Via Probability Map Guided Copy-Pasting"
    Implementation details can refer to https://github.com/GothicAi/Instaboost.
    """

    def __init__(self,
                 action_candidate=('normal', 'horizontal', 'skip'),
                 action_prob=(1, 0, 0),
                 scale=(0.8, 1.2),
                 dx=15,
                 dy=15,
                 theta=(-1, 1),
                 color_prob=0.5,
                 hflag=False,
                 aug_ratio=0.5,
                 resize_scale=None,
                 max_instance_num=10000):

        self.cfg = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                               scale, dx, dy, theta,
                                               color_prob, hflag)
        self.aug_ratio = aug_ratio

        self.scale = resize_scale
        self.max_instance_num = max_instance_num

    def _load_anns(self, results):
        labels = results['ann_info']['labels']
        masks = results['ann_info']['masks']
        bboxes = results['ann_info']['bboxes']
        n = len(labels)

        anns = []
        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            mask = masks[i]
            x1, y1, x2, y2 = bbox
            # assert (x2 - x1) >= 1 and (y2 - y1) >= 1
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append({
                'category_id': label,
                'segmentation': mask,
                'bbox': bbox
            })

        return anns

    def _parse_anns(self, results, anns, img):
        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            # TODO: more essential bug need to be fixed in instaboost
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            gt_masks_ann.append(ann['segmentation'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        results['ann_info']['labels'] = gt_labels
        results['ann_info']['bboxes'] = gt_bboxes
        results['ann_info']['masks'] = gt_masks_ann
        results['img'] = img
        return results

    def _resize_img(self, results_in):
        """Resize images with ``results['scale']``."""
        for key in results_in.get('img_fields', ['img']):
            img, w_scale, h_scale = mmcv.imresize(
                results_in[key],
                self.scale,
                return_scale=True)
            results_in[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results_in['img_shape'] = img.shape
            # in case that there is no padding
            results_in['pad_shape'] = img.shape
            self.scale_factor = scale_factor

        return results_in

    def _resize_bboxes(self, results_in):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results_in['img_shape']

        results_in['ann_info']['bboxes'] = results_in['ann_info']['bboxes'] * self.scale_factor
        results_in['ann_info']['bboxes'][:, 0::2] = np.clip(results_in['ann_info']['bboxes'][:, 0::2],
                                                            0, img_shape[1])
        results_in['ann_info']['bboxes'][:, 1::2] = np.clip(results_in['ann_info']['bboxes'][:, 1::2],
                                                            0, img_shape[0])

        return results_in

    def _cocoseg_poly_to_binary(self, seg, height, width):
        """
        COCO style segmentation to binary mask
        :param seg: coco-style segmentation
        :param height: image height
        :param width: image width
        :return: binary mask
        """
        if type(seg) == list:
            rle = cocomask.frPyObjects(seg, height, width)
            rle = cocomask.merge(rle)
            mask = cocomask.decode([rle])
            assert mask.shape[2] == 1
            return mask[:, :, 0]

    def _switch_order(self, item):
        import copy
        first = [item.pop(0)]
        tmp = copy.deepcopy(item)
        for each_item in tmp:
            if len(each_item) <= 4:
                first += [item.pop(0)]
            else:
                item += first
                return item

        return item

    def _check_mask(self, results_in):
        need_conversion = 0
        if 'masks' not in results_in['ann_info']:
            need_conversion = 1
        elif len(results_in['ann_info']['masks']) == 0:
            need_conversion = 1
        elif len(results_in['ann_info']['masks'][0]) == 0:
            need_conversion = 1

        if need_conversion:
            results_in['ann_info']['masks'] = []
            for idx, item in enumerate(results_in['ann_info']['bboxes']):
                tmp = [item[0], item[1], item[0]+item[2], item[1], item[0]+item[2],
                       item[1]+item[3], item[0], item[1] + item[3]]
                results_in['ann_info']['masks'].append([tmp])

        return results_in

    def _remove_small_masks(self, results_in):
        null_idx = []
        for idx, item in enumerate(results_in['ann_info']['masks']):
            if len(item) == 0:
                null_idx.append(idx)
            elif len(item[0]) <= 4:
                item = self._switch_order(item)
                if len(item) == 0:
                    null_idx.append(idx)

        if len(null_idx) > 0:
            tmp_mask = np.array(results_in['ann_info']['masks'], dtype=object)
            results_in['ann_info']['masks'] = list(np.delete(tmp_mask, null_idx, axis=0))
            results_in['ann_info']['bboxes'] = np.delete(results_in['ann_info']['bboxes'], null_idx, axis=0)
            results_in['ann_info']['labels'] = np.delete(results_in['ann_info']['labels'], null_idx, axis=0)

        return results_in

    def _resize_masks(self, results_in):
        """Resize masks with ``results['scale']``"""
        for result_idx, items in enumerate(results_in['ann_info']['masks']):
            temp = []
            if type(items) == list:
                # polygon
                inst_mask_in = self._cocoseg_poly_to_binary(items, results_in['img_info']['height'],
                                                            results_in['img_info']['width'])
                inst_mask, w_scale, h_scale = mmcv.imresize(inst_mask_in, self.scale, return_scale=True)
                fortran = np.asfortranarray(inst_mask)
                contours = measure.find_contours(fortran, 0.5)
                for contour in contours:
                    ct = np.flip(contour, axis=1)
                    segmentation = ct.ravel().tolist()
                    temp.append(segmentation)

            results_in['ann_info']['masks'][result_idx] = temp

        results_in = self._remove_small_masks(results_in)
        results_in['img_info']['height'] = results_in['img'].shape[0]
        results_in['img_info']['width'] = results_in['img'].shape[1]

        return results_in

    def _preprocess(self, results_in):
        # if the input is larger than the target size, resize the input to the target size
        results_in = self._resize_img(results_in)
        results_in = self._resize_bboxes(results_in)
        results_in = self._resize_masks(results_in)
        img = results_in['img']
        anns = self._load_anns(results_in)

        return anns, img, results_in

    def __call__(self, results):
        import copy
        tmp_results = copy.deepcopy(results)
        img = tmp_results['img']
        orig_type = img.dtype
        tmp_results = self._check_mask(tmp_results)
        if type(tmp_results['ann_info']['masks'][0]) == dict:
            raise ValueError('You should not input json containing RLE annotations for \
                                         input resize @ instaboost.py!')
        tmp_results = self._remove_small_masks(tmp_results)
        anns = self._load_anns(tmp_results)
        if np.random.choice([0, 1], p=[1 - self.aug_ratio, self.aug_ratio]):
            # resize input
            if self.scale is not None:
                tmp_h = tmp_results['img_info']['height']
                tmp_w = tmp_results['img_info']['width']
                if tmp_h * tmp_w > self.scale[1] * self.scale[0]:
                    anns, img, tmp_results = self._preprocess(tmp_results)
            # set maximum # of instances
            if len(anns) > self.max_instance_num:
                area = []
                for items in anns:
                    area.append(items['bbox'][2]*items['bbox'][3])
                sorted_idx = [i[0] for i in sorted(enumerate(area), key=lambda x:x[1], reverse=True)]
                anns_temp2 = np.array(anns)
                anns_temp = list(anns_temp2[sorted_idx[0:self.max_instance_num]])
                anns_temp2 = np.delete(anns_temp2, sorted_idx[0:self.max_instance_num])
                anns = list(anns_temp2)
                anns_temp, img = instaboost.get_new_data(
                    anns_temp, img.astype(np.uint8), self.cfg, background=None)

                anns.extend(anns_temp)
            else:
                anns, img = instaboost.get_new_data(
                    anns, img.astype(np.uint8), self.cfg, background=None)
        else:
            if self.scale is not None:
                tmp_h = tmp_results['img_info']['height']
                tmp_w = tmp_results['img_info']['width']
                if tmp_h * tmp_w > self.scale[1] * self.scale[0]:
                    anns, img, tmp_results = self._preprocess(tmp_results)
        tmp_results = self._parse_anns(tmp_results, anns, img.astype(orig_type))

        return tmp_results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(cfg={self.cfg}, aug_ratio={self.aug_ratio})'
        return repr_str

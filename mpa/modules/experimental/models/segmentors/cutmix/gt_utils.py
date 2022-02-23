import numpy as np
import random
import torch
from skimage.measure import label, regionprops


def padding_bbox_new(rectangles, size):
    area = 0.5 * (size ** 2)
    y0, x0, y1, x1 = rectangles
    h = y1 - y0
    w = x1 - x0
    # upper_h = min(int(area/w), size)
    # upper_w = min(int(area/h), size)
    new_h = int(size*(np.exp(np.random.uniform(low=0.0, high=1.0, size=(1)) * np.log(0.5))))
    new_w = int(area/new_h)
    delta_h = new_h - h
    delta_w = new_w - w
    y_ratio = y0/(size-y1+1)
    x_ratio = x0/(size-x1+1)
    x1 = min(x1+int(delta_w*(1/(1+x_ratio))), size)
    x0 = max(x0-int(delta_w*(x_ratio/(1+x_ratio))), 0)
    y1 = min(y1+int(delta_h*(1/(1+y_ratio))), size)
    y0 = max(y0-int(delta_h*(y_ratio/(1+y_ratio))), 0)
    return [y0, x0, y1, x1]


def sliming_bbox(rectangles, size):
    area = 0.5 * (size ** 2)
    y0, x0, y1, x1 = rectangles
    h = y1 - y0
    w = x1 - x0
    lower_h = int(area/w)
    if lower_h > h:
        # print('wrong')
        new_h = h
    else:
        new_h = random.randint(lower_h, h)
    new_w = int(area/new_h)
    if new_w > w:
        # print('wrong')
        new_w = w - 1
    delta_h = h - new_h
    delta_w = w - new_w
    prob = random.random()
    if prob > 0.5:
        y1 = max(random.randint(y1 - delta_h, y1), y0)
        y0 = max(y1 - new_h, y0)
    else:
        y0 = min(random.randint(y0, y0 + delta_h), y1)
        y1 = min(y0 + new_h, y1)
    prob = random.random()
    if prob > 0.5:
        x1 = max(random.randint(x1 - delta_w, x1), x0)
        x0 = max(x1 - new_w, x0)
    else:
        x0 = min(random.randint(x0, x0 + delta_w), x1)
        x1 = min(x0 + new_w, x1)
    return [y0, x0, y1, x1]


def init_cutmix(crop_size):
    h = crop_size
    w = crop_size
    n_masks = 1
    prop_range_from = 0.15
    prop_range = 0.4
    mask_props = np.random.uniform(prop_range_from, prop_range, size=(n_masks, 1))
    y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(n_masks, 1)) * np.log(mask_props))
    x_props = mask_props / y_props
    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :])
    positions = np.round((np.array((h, w))-sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
    rectangles = np.append(positions, positions+sizes, axis=2)[0, 0]

    return rectangles


def generate_cutmix_mask(pred, sample_cat, area_thresh=0.0001, no_pad=False, no_slim=False):
    h, w = pred.shape[2], pred.shape[3]
    valid_masks = np.zeros_like(pred)
    n_boxes = 1
    for each_image_in_batch in range(pred.shape[0]):
        valid_mask = np.zeros((h, w))
        values = list(np.unique(pred[each_image_in_batch][0]))
        for i in range(n_boxes):
            if not sample_cat[each_image_in_batch] in values or i > 0:
                rectangles = init_cutmix(h)
                y0, x0, y1, x1 = rectangles
                valid_mask[int(y0):int(y1), int(x0):int(x1)] = 1 - valid_mask[int(y0):int(y1), int(x0):int(x1)]
            else:
                rectangles = generate_cutmix(pred[each_image_in_batch][0], sample_cat[each_image_in_batch],
                                             area_thresh, no_pad=no_pad, no_slim=no_slim)
                y0, x0, y1, x1 = rectangles
                tmp = [[int(y0), int(y1/2), int(x0), int(x1/2)],
                       [int(y0), int(y1/2), int(x1/2), int(x1)],
                       [int(y1/2), int(y1), int(x0), int(x1/2)],
                       [int(y1/2), int(y1), int(x1/2), int(x1)]]
                # tmp_cnt = 0
                for i in range(4):
                    if np.random.rand() < 0.5:
                        # tmp_cnt += 1
                        val_mask = valid_mask[tmp[i][0]:tmp[i][1], tmp[i][2]:tmp[i][3]]
                        valid_mask[tmp[i][0]:tmp[i][1], tmp[i][2]:tmp[i][3]] = 1 - val_mask

        # valid_mask[int(y0):int(y1), int(x0):int(x1)] = 1
        valid_masks[each_image_in_batch][0] = torch.from_numpy(valid_mask).long()

    # if pred.is_cuda:
    #     valid_masks = valid_masks.cuda()

    return torch.Tensor(valid_masks)


def generate_cutmix(pred, cat, area_thresh, no_pad=False, no_slim=False):
    h = pred.shape[0]
    # print('h',h)
    area_all = h ** 2
    pred = (pred == cat) * 1
    pred = label(pred)
    prop = regionprops(pred)
    values = np.unique(pred)[1:]
    random.shuffle(values)

    flag = 0
    for value in values:
        if np.sum(pred == value) > area_thresh*area_all:
            flag = 1
            break
    if flag == 1:
        rectangles = prop[value-1].bbox
        # area = prop[value-1].area
        area = (rectangles[2]-rectangles[0])*(rectangles[3]-rectangles[1])
        if area >= 0.5*area_all and not no_slim:
            rectangles = sliming_bbox(rectangles, h)
        elif area < 0.5*area_all and not no_pad:
            rectangles = padding_bbox_new(rectangles, h)
        else:
            pass
    else:
        rectangles = init_cutmix(h)
    return rectangles


def update_cutmix_bank(cutmix_bank, preds_teacher_unsup, img_id, sample_id, area_thresh=0.0001):
    # cutmix_bank [num_classes, len(dataset)]
    # preds_teacher_unsup [2,num_classes,h,w]
    area_all = preds_teacher_unsup.shape[-1]**2
    pred1 = preds_teacher_unsup[0].max(0)[1]   # (h,w)
    pred2 = preds_teacher_unsup[1].max(0)[1]   # (h,w)
    values1 = torch.unique(pred1)
    values2 = torch.unique(pred2)
    # for img1
    for idx in range(cutmix_bank.shape[0]):
        if idx not in values1:
            cutmix_bank[idx][img_id] = 0
        elif torch.sum(pred1 == idx) < area_thresh * area_all:
            cutmix_bank[idx][img_id] = 0
        else:
            cutmix_bank[idx][img_id] = 1
    # for img2
    for idx in range(cutmix_bank.shape[0]):
        if idx not in values2:
            cutmix_bank[idx][sample_id] = 0
        elif torch.sum(pred2 == idx) < area_thresh * area_all:
            cutmix_bank[idx][sample_id] = 0
        else:
            cutmix_bank[idx][sample_id] = 1

    return cutmix_bank

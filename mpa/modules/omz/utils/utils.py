import os

import numpy as np
import torch
from torch.autograd import Variable

# from tlt.core.omz.data.voc_data import COLORS, VOC_CLASSES
# from detection.boxes_utils import clip_boxes, bbox_transform_inv, py_cpu_nms
# from detection.boxes_utils import *


def to_var(x, *args, **kwargs):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if torch.cuda.is_available():
        x = Variable(x, *args, **kwargs).cuda()
    else:
        x = Variable(x, *args, **kwargs).cpu()

    return x


def to_tensor(x, *args, **kwargs):

    if torch.cuda.is_available():
        x = torch.from_numpy(x).cuda()
    else:
        x = torch.from_numpy(x)
    return x


def make_name_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]


def read_pickle(path, model, solver):

    try:
        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/model_" + recent_iter + ".pkl", "rb") as f:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(f))
            else:
                model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

        with open(path + "/solver_" + recent_iter + ".pkl", "rb") as f:
            if torch.cuda.is_available():
                solver.load_state_dict(torch.load(f))
            else:
                solver.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    except Exception as e:

        print("fail try read_pickle", e)


def save_pickle(path, epoch, model, solver):

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/model_" + str(epoch) + ".pkl", "wb") as f:
        torch.save(model.state_dict(), f)
    with open(path + "/solver_" + str(epoch) + ".pkl", "wb") as f:
        torch.save(solver.state_dict(), f)


def save_image(path, iteration, img):

    if not os.path.exists(path):
        os.makedirs(path)

    img.save(path + "/img_" + str(iteration) + ".png")

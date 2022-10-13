from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from abc import abstractmethod


class BaseExplainer(ABC):
    def __init__(self, model):
        self._model = model
    
    @abstractmethod
    def get_saliency_map(self):
        pass

    @abstractmethod
    def run(self):
        pass

# RISE: Randomized Input Sampling for Explanation of Black-box Models
# https://arxiv.org/pdf/1806.07421.pdf
# refer from https://github.com/yiskw713/RISE/blob/master/rise.py

class RISE(nn.Module):
    def __init__(self, model, n_masks=10000, p1=0.1, input_size=(224, 224), initial_mask_size=(7, 7), n_batch=128, mask_path=None):
        super().__init__()
        self.model = model
        self.n_masks = n_masks
        self.p1 = p1
        self.input_size = input_size
        self.initial_mask_size = initial_mask_size
        self.n_batch = n_batch

        if mask_path is not None:
            self.masks = self.load_masks(mask_path)
        else:
            self.masks = self.generate_masks()

    def generate_masks(self):
        # cell size in the upsampled mask
        Ch = np.ceil(self.input_size[0] / self.initial_mask_size[0])
        Cw = np.ceil(self.input_size[1] / self.initial_mask_size[1])

        resize_h = int((self.initial_mask_size[0] + 1) * Ch)
        resize_w = int((self.initial_mask_size[1] + 1) * Cw)

        masks = []

        for _ in range(self.n_masks):
            # generate binary mask
            binary_mask = torch.randn(
                1, 1, self.initial_mask_size[0], self.initial_mask_size[1])
            binary_mask = (binary_mask < self.p1).float()

            # upsampling mask
            mask = F.interpolate(
                binary_mask, (resize_h, resize_w), mode='bilinear', align_corners=False)

            # random cropping
            i = np.random.randint(0, Ch)
            j = np.random.randint(0, Cw)
            mask = mask[:, :, i:i+self.input_size[0], j:j+self.input_size[1]]

            masks.append(mask)

        masks = torch.cat(masks, dim=0)   # (N_masks, 1, H, W)

        return masks

    def load_masks(self, filepath):
        masks = torch.load(filepath)
        return masks

    def save_masks(self, filepath):
        torch.save(self.masks, filepath)

    def forward(self, x):
        # x: input image. (1, 3, H, W)
        device = x.device

        # keep probabilities of each class
        probs = []
        # shape (n_masks, 3, H, W)
        masked_x = torch.mul(self.masks, x.to('cpu').data)

        for i in range(0, self.n_masks, self.n_batch):
            input = masked_x[i:min(i + self.n_batch, self.n_masks)].to(device)
            out = self.model(input)
            probs.append(torch.softmax(out, dim=1).to('cpu').data)

        probs = torch.cat(probs)    # shape => (n_masks, n_classes)
        n_classes = probs.shape[1]

        # caluculate saliency map using probability scores as weights
        saliency = torch.matmul(
            probs.data.transpose(0, 1),
            self.masks.view(self.n_masks, -1)
        )
        saliency = saliency.view(
            (n_classes, self.input_size[0], self.input_size[1]))
        saliency = saliency / (self.n_masks * self.p1)

        # normalize
        m, _ = torch.min(saliency.view(n_classes, -1), dim=1)
        saliency -= m.view(n_classes, 1, 1)
        M, _ = torch.max(saliency.view(n_classes, -1), dim=1)
        saliency /= M.view(n_classes, 1, 1)
        return saliency.data
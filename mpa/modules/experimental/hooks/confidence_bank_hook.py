import torch
import torch.nn.functional as F

from mmcv.runner import HOOKS, Hook
# from mmcv.parallel import is_module_wrapper


@HOOKS.register_module()
class ConfidenceBankHook(Hook):
    def __init__(self,
                 class_momentum=0.999,
                 num_classes=21,
                 **kwargs):
        super().__init__(**kwargs)

        self.class_momentum = class_momentum
        self.num_classes = num_classes

        print('####################### ConfidenceBankHook')

    def after_train_iter(self, runner):
        logits = runner.outputs['logits'].detach()
        logits = F.softmax(logits, dim=1)
        gt = runner.outputs['gt_semantic_seg'].detach()
        prev_conf = runner.data_loader._dataloader.dataset.class_criterion

        category_entropy = self.cal_category_confidence(logits, gt)
        curr_conf = prev_conf * self.class_momentum + category_entropy * (1 - self.class_momentum)

        # runner.data_loader._dataloader.dataset.update_confidence_bank(curr_conf)
        runner.data_loader.iter_loader._dataset.update_confidence_bank(curr_conf)

    def cal_category_confidence(self, logits, gt):
        category_confidence = torch.zeros(self.num_classes).type(torch.float32)
        logits = F.softmax(logits, dim=1)
        for idx in range(self.num_classes):
            mask_cls = (gt == idx)
            if torch.sum(mask_cls) == 0:
                value = 0
            else:
                conf_map_sup = logits[:, idx, :, :]
                value = torch.sum(conf_map_sup * mask_cls) / (torch.sum(mask_cls) + 1e-12)

            category_confidence[idx] = value

        return category_confidence

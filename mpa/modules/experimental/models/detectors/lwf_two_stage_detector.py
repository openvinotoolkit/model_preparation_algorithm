from mmdet.models import DETECTORS
from mpa.modules.models.detectors.custom_two_stage_detector import CustomTwoStageDetector
import torch


@DETECTORS.register_module()
class LwfTwoStageDetector(CustomTwoStageDetector):
    """LwF-enabled 2-stage detector
    """

    def __init__(self, arch, src_classes, dst_classes, **kwargs):
        print('LwF-2-stage-detector initilaized!')
        super().__init__(**kwargs)
        self.arch = arch
        self.src_classes = src_classes
        self.dst_classes = dst_classes

        # Initialize teacher model (for OLD classes)
        roi_head_cfg = kwargs.pop('roi_head', None)
        if roi_head_cfg:
            roi_head_cfg['bbox_head']['num_classes'] = len(src_classes)
        teacher = CustomTwoStageDetector(roi_head=roi_head_cfg, **kwargs)
        # - Weight fixed -> 'evaluate' mode + no gradient
        teacher.eval()
        # - Keep in array to prevent checkpointing teacher model
        self.teachers = [teacher]
        self._teacher = teacher  # HOTFIX for model-level cuda()/train()/eval()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """Learning w/o Forgetting
        """
        teacher = self.teachers[0]
        teacher.eval()

        # Feature
        with torch.no_grad():
            x_t = teacher.extract_feat(img)
        x = self.extract_feat(img)

        losses = dict()

        # Region proposal & losses
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # Teacher (OLD tasks)
            with torch.no_grad():
                rpn_outputs_t = teacher.rpn_head(x_t)

            # Student
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                teacher_rpn_outputs=rpn_outputs_t,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # ROI outputs & losses
        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            x_t=x_t,  # teacher feature map for LwF inside
            roi_head_t=teacher.roi_head,  # teacher roi_head for LwF inside
            **kwargs)
        losses.update(roi_losses)

        if self.l2sp:
            losses.update(dict(loss_l2sp=self.l2sp()))
        return losses

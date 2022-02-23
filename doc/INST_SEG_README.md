# Transfer Learning for Instance Segmentation

## Introduction
In practical cases, users generally have the limited number of training images, thus it is hard to achieve high prediction accuracy.
Instance-level data augmentation is one of popular methods to improve the performance of the model trained with the limited number of training samples,
but its computation burden is high and it requires longer epoch than the baseline method.
To solve this problem, we profile instance-level data augmentation and disable inefficient modules, thus supporting the efficient transfer learning recipe for instance segmentation.

## How to use?
- Basic usage of the custom instance segmentation recipes
  - To train custom instance segmentation, the base model needs to be prepared. If you don't have a base model, we highly recommand you to download the COCO pre-trained models from OpenVino Training Extension web-site (https://github.com/openvinotoolkit/training_extensions/tree/develop/models/instance_segmentation/model_templates/coco-instance-segmentation).
    - The default model is Efficient-b2b and its pretrained model is (https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/instance_segmentation/v2/instance-segmentation-1039.pth)
    - if you have the pretrained model, go to "Fine-tuning from the pre-trained model" step
  - The trained model (checkpoint, *pth) is stored in {{output_dir}}/{{stage name described in recipe.yaml}}
  - Usage
    ```bash
    (mpa) .../mpa$ python tools/cli.py {recipe, *.yaml file} --model_cfg {model, *.yaml file} --data_cfg {data, *.yaml file} --output_path {output_dir}
    ```
  - Example
    ```bash
    (mpa) .../mpa$ python tools/cli.py recipes/segmentation/domain_adapt/instance_segm/finetune/recipe.yaml --model_cfg models/detection/mask_rcnn_eff_b2b_fpn_1x_coco.py --data_cfg samples/cfgs/inst_seg/coco_repeated.data.yaml --output_path ./logs
    ```
  - Example of recipe.yaml
    - [recipe.yaml](../recipes/segmentation/domain_adapt/instance_segm/finetune/recipe.yaml)
  - Example of model.yaml
    - [model.yaml](../models/detection/mask_rcnn_eff_b2b_fpn_1x_coco.py)
  - Example of data.yaml
    - [data.yaml](../samples/cfgs/inst_seg/coco_repeated.data.yaml)

- Fine-tuning from the pre-trained model
  - From the pre-trained model on target datasets, we can boost the performance by training some more epoch with instance level augmentation.
  - Usage
    ```bash
    (mpa) .../mpa$ python tools/cli.py {recipe, *.yaml file} --model_cfg {model, *.yaml file} --data_cfg {data, *.yaml file} --output_path {output_dir} --model_ckpt {*.pth file}
    ```
  - Example
    ```bash
    (mpa) .../mpa$ python tools/cli.py recipes/segmentation/domain_adapt/instance_segm/finetune/recipe.yaml --model_cfg models/detection/mask_rcnn_eff_b2b_fpn_1x_coco.py --data_cfg samples/cfgs/inst_seg/coco_repeated.data.yaml --output_path ./logs --model_ckpt models/detection/instance-segmentation-1039.pth
    ```
  - Example of recipe.yaml
    - [recipe.yaml](../recipes/segmentation/domain_adapt/instance_segm/finetune/recipe.yaml)
  - Example of model.yaml
    - [model.yaml](../models/detection/mask_rcnn_eff_b2b_fpn_1x_coco.py)
  - Example of data.yaml
    - [data.yaml](../samples/cfgs/inst_seg/coco_repeated.data.yaml)

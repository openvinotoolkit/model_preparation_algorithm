# Class Incremental Learning for Classification

## Introduction
**Class incremental learning for classification** is one of transfer learning methods to train new classes while preserving the accuracy of the previously trained classes.
Currently, we support the learning without forgetting (LwF) based method.
Since LwF based multi-class classification does not use the data from previously trained class, 
it is memory-efficient and fast to train the model.

## How to use?
- Preparation for the base model
  - The base model for class incremental learning needs to be prepared. If you don't have a base model, you need to train it first.
  - You can fine-tune the base model from previoulsy trained model, or train it from scratch.      
  - The trained model (checkpoint, *pth) is stored in {{output_dir}}/{{stage name described in recipe.yaml}}
  - Usage
    ```bash   
    (mpa) .../mpa$ python tools/cli.py {recipe, *.yaml file} --model_cfg {model, *.yaml file} --data_cfg {data, *.yaml file} --output_path {output_dir} --model_ckpt {*.pth file} 
    ```
  - Example
    ```bash   
    (mpa) .../mpa$ python tools/cli.py recipes/classification/domain_adapt/finetune/recipe.yaml --model_cfg samples/cfgs/single_task/single_task.model.yaml --data_cfg samples/cfgs/single_task/single_task.data100.yaml --output_path ./logs
    ```
  - Example of recipe.yaml
    - [recipe.yaml](../recipes/classification/domain_adapt/finetune/recipe.yaml) 
  - Example of model.yaml
    - [model.yaml](../samples/cfgs/single_task/single_task.model.yaml) 
  - Example of data.yaml      
    - [data.yaml](../samples/cfgs/single_task/single_task.data100.yaml) 

- Class incremental learning from the base model
  - From the base model, we can add new classes in data.yaml file as follow:
    ```
      train:
        type: 'TVDatasetSplitClsInc'  
        base: 'CIFAR100'  
        data_prefix: 'data/torchvision/cifar100'
        download: True
        num_of_old_class: 5
        num_classes: 5    
        classes: [1,6,11,16,21]
    ```
    - base : the recipe supports the torchvision datasets, thus 'base' should be one of the torchvision datasets.
    - num_of_old_class: describe how many classes are trained in the base model.
    - num_classes: describe how many classes will be added.
    - classes: describe the classe name which will be added.
    ```
      val:
        type: 'TVDatasetSplitClsInc'    
        base: 'CIFAR100'
        data_prefix: 'data/torchvision/cifar100'
        download: True
        num_classes: 10
        classes: [0,5,10,15,20,1,6,11,16,21]   
    ```    
    - num_classes: describe how many classes will be validated.
    - classes: describe the classe name which will be validated.
    - Usage for 'test mode' is the same to 'val. mode'.
  - The trained model (checkpoint, *pth) is stored in {{output_dir}}/{{stage name described in recipe.yaml}}
  - Usage
    ```bash   
    (mpa) .../mpa$ python tools/cli.py {recipe, *.yaml file} --model_cfg {model, *.yaml file} --data_cfg {data, *.yaml file} --output_path {output_dir} --model_ckpt {*.pth file} 
    ```
  - Example
    ```bash   
    (mpa) .../mpa$ python tools/cli.py recipes/classification/task_adapt/incremental/cls_LwF/recipe.yaml --model_cfg samples/cfgs/cls_inc/cls_inc.model.yaml --data_cfg samples/cfgs/cls_inc/cls_inc.data100.yaml --model_ckpt tests/assets/model_cfg/mobilenetv2_cls_inc/best_model.pth --output_path ./logs
    ```
    - Example of recipe.yaml
      - [recipe.yaml](../recipes/classification/task_adapt/incremental/cls_LwF/recipe.yaml) 
    - Example of model.yaml
      - [model.yaml](../samples/cfgs/cls_inc/cls_inc.model.yaml) 
    - Example of data.yaml      
      - [data.yaml](../samples/cfgs/cls_inc/cls_inc.data100.yaml) 
    - Example of pre-trained model
      - The "best_model" is trained by using 0,5,10,15,20-th classes of CIFAR-100 dataset.

# Model drift analysis

## Introduction
**Model drift analysis** measures discrepancy beytween model and dataset providing you discrepancy score. The discrepancy score shows how well model is fitted to dataset quantitatively.

## How to use?
MDA exists as an independent stage. So you need to add MDA stage which set 'type' value to 'MdaRunner' and MDA configurations in recipe file. Of course, both model and dataset to measure discrepancy should be set by CLI arguments or recipe file as well.  
MDA configuration lists are as below.  
* **task** : 'classification' or 'detection'. What type of task model works for.
* **mda_metric** (optional) : 'z-score', 'cos-sim', 'kl' or 'wst'. Metric of MDA. Default value is 'z-score'  

You can easily run MDA stage just by modifying 'type' value to 'MdaRunner' in any general classification or detection recipe.yaml file.  
Example of config.yaml file format is as below.
```
stages:
  - name: finetune
    type: MdaRunner
    task: classification
    mode: train
    config: {{ fileDirname }}/train.yaml
    mda_metric: z-score
    output:
      - final_score
```
You can also find example recipe.yaml files in 
* recipes/classification/domain_adapt/finetune/mda_recipe.yaml (classification)
* recipes/detection/domain_adapt/faster_rcnn/finetune/mda_recipe.yaml (detection)

## Output
Discrepancy score is displayed in terminal and result file named *mda_result.txt* is saved in configured *work_dir* after MDA stage is done.  
Results are like as below after MDA stage run.
```bash
...
Number of batch-norm layers : 52
100%|████████████████████████████████████████████| 469/469 [00:22<00:00, 20.83it/s]
Average discrepancy :  26.063466608103955
(mpa) .../mpa/output_dir/stage00_stagename$ ls
...
mda_result.txt
```

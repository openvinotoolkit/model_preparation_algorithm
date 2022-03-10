#!/bin/bash
OTE_ROOT=/home/jisunkwo/workspace/omz/training_extensions

# copy and download the ote model
OTE_MODEL=person-detection-0200
python ${OTE_ROOT}/pytorch_toolkit/tools/instantiate_template.py ${OTE_ROOT}/pytorch_toolkit/object_detection/model_templates/person-detection/${OTE_MODEL}/template.yaml ./ote_models/${OTE_MODEL}/

ln -srf ./ote_models/${OTE_MODEL}/model.py ./ote_example/ote_model.py

python -m tools.cli ./recipes/ote/recipes_byol_otedet.yaml --model_cfg ./ote_models/${OTE_MODEL}/model.py --data_cfg ./ote_example/data_coco_taskadapt_car.yaml --model_ckpt ./ote_models/${OTE_MODEL}/snapshot.pth --output_path outputs/${OTE_MODEL}

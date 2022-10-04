#!/bin/bash
dataset="fish"
subset="16"
trial="1"

python tools/cli.py ./recipes/experimental/det_default.yaml \
--data_cfg=./samples/cfgs/data/experimental/${dataset}_${subset}_${trial}.py \
--output_path=./logs/temp_det
CUDA_VISIBLE_DEVICES=2 python tools/cli.py \
./recipes/cls_class_incr.yaml \
--model_cfg ./models/classification/_base_/mobilenet_v2.py \
--output_path work_dirs/explain \
--seed 42 \
--mode explain \
--recipe_hparams data.num_classes=10 \
--recipe_hparams data.samples_per_gpu=512 \
--recipe_hparams data.workers_per_gpu=16 \
data.train.data_dir=/home/dongkwan/training_extensions/data/classification/train \
data.val.data_dir=/home/dongkwan/training_extensions/data/classification/train \
data.test.data_dir=/home/dongkwan/training_extensions/data/classification/train \
# data.train.data_dir=/mnt/storageserver/fd_fa_data/datasets/cifar10/train \
# data.val.data_dir=/mnt/storageserver/fd_fa_data/datasets/cifar10/test \
# data.test.data_dir=/mnt/storageserver/fd_fa_data/datasets/cifar10/test \


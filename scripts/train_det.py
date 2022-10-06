import os
import subprocess


GPUS = 0
DATASET = 'fish'
DATAROOT = f'data/{DATASET}'

MODELS = ['detcon_mnv2_atss']
NUM_DATAS = [16]
BATCHSIZES = [8]
INTERVALS = [1]
LAMBDAS = [1]

if DATASET == 'fish':
    REAL_CLASSES = '"["fish"]"'
    NUM_CLASSES = 1
elif DATASET == 'bccd':
    REAL_CLASSES = '"["WBC", "RBC", "Platelets"]"'
    NUM_CLASSES = 3
elif DATASET == 'pothole':
    REAL_CLASSES = '"["pothole"]"'
    NUM_CLASSES = 1
elif DATASET == 'vitens':
    REAL_CLASSES = '"["vitens"]"'
    NUM_CLASSES = 1
else:
    raise ValueError()

for MODEL in MODELS:
    for NUM_DATA in NUM_DATAS:
        for BATCHSIZE in BATCHSIZES:
            for INTERVAL in INTERVALS:
                for LAMBDA in LAMBDAS:
                    for seed in [1]:
                        # set command
                        if 'detcon' in MODEL:
                            PARAMS_MODEL = (
                                f'hyperparams.model.num_classes={NUM_CLASSES} '
                                # f'hyperparams.model.loss_weights.detcon={LAMBDA} '
                                # f'hyperparams.custom_hooks.1.interval={INTERVAL} '
                            )
                            WORKDIR = f'work_dirs/supcon/detection/{DATASET}/{NUM_DATA}/asis_lambda{LAMBDA}_batch{BATCHSIZE}_interval{INTERVAL}/{MODEL}_seed{seed}'
                        else:
                            assert len(LAMBDAS) == 1 and len(INTERVALS) == 1
                            PARAMS_MODEL = ''
                            WORKDIR = f'work_dirs/supcon/detection/{DATASET}/{NUM_DATA}/asis_batch{BATCHSIZE}/{MODEL}_seed{seed}'

                        RECIPE = f'./recipes/experimental/det_{MODEL}.yaml '
                        os.makedirs(WORKDIR, exist_ok=True)
                        command = (
                            f'CUDA_VISIBLE_DEVICES={GPUS} python tools/cli.py '
                                f'{RECIPE}'
                                f'--output_path {WORKDIR} '
                                f'--recipe_hparams '
                                f'hyperparams.data.samples_per_gpu={BATCHSIZE} '
                                f'hyperparams.data.train.ann_file=data/{DATASET}/annotations/instances_train_{NUM_DATA}_{seed}.json '
                                f'hyperparams.data.train.img_prefix=data/{DATASET}/images/train/ '
                                f'hyperparams.data.train.classes={REAL_CLASSES} '
                                f'hyperparams.data.val.ann_file=data/{DATASET}/annotations/instances_val_100.json '
                                f'hyperparams.data.val.img_prefix=data/{DATASET}/images/val/ '
                                f'hyperparams.data.val.classes={REAL_CLASSES} '
                                f'hyperparams.data.test.ann_file=data/{DATASET}/annotations/instances_val_100.json '
                                f'hyperparams.data.test.img_prefix=data/{DATASET}/images/val/ '
                                f'hyperparams.data.test.classes={REAL_CLASSES} '
                        )
                        command += PARAMS_MODEL
                        command += f'2>&1 | tee {WORKDIR}/output.log'

                        subprocess.run(command, shell=True)

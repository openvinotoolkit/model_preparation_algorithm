import argparse
import json

# workaround to avoid ops kernel registry conflict
# from mmcv import ops  # noqa: F401

from mmcv import DictAction

from mpa.builder import build
from mpa.stage import Stage, get_available_types
from mpa.utils.config_utils import MPAConfig


def main(args):
    print(args)
    # output_path = args.output_path

    # Model config
    model_cfg = MPAConfig.fromfile(args.model_cfg) if args.model_cfg is not None else None

    # Data config
    data_cfg = MPAConfig.fromfile(args.data_cfg) if args.data_cfg is not None else None

    # Recipe config
    recipe_cfg = MPAConfig.fromfile(args.recipe_cfg)

    if args.recipe_hparams:
        recipe_cfg.merge_from_dict(args.recipe_hparams)

    if args.recipe_json is not None:
        recipe_cfg.marge_from_dict(json.loads(args.recipe_json))

    if args.log_level != 'none':
        # setattr(recipe_cfg, 'log_level', args.log_level)
        recipe_cfg.log_level = args.log_level
        # if hasattr(recipe_cfg, 'log_level'):
        #     recipe_cfg.log_level = args.log_level

    # configure output path prefix
    recipe_cfg.output_path = args.output_path

    # Mode
    mode = args.mode

    # Workflow
    recipe = build(recipe_cfg, mode, stage_type=args.stage_type)
    if isinstance(recipe, Stage):
        recipe.run(
            stage_idx=0,
            mode=mode,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            ir_path=args.ir_path,
            model_ckpt=args.model_ckpt
            )
    else:
        recipe.run(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            ir_path=args.ir_path,
            model_ckpt=args.model_ckpt,
            # output_path=args.output_path,
            mode=mode
        )


if __name__ == "__main__":
    get_available_types()
    parser = argparse.ArgumentParser(description="TL-Recipe Command Line Interface")
    parser.add_argument('recipe_cfg', type=str,
                        help='Path to config file of transfer learning recipe')
    parser.add_argument('--model_cfg', type=str,
                        help='Path to config file of (base) model for train/infer/eval')
    parser.add_argument('--ir_path', type=str,
                        help='Path to XML file of OMZ model to load')
    parser.add_argument('--model_ckpt', type=str,
                        help='Path to checkpoint file to load')
    parser.add_argument('--data_cfg', type=str,
                        help='Path to config file for input data. root/file/type for each \
                              train/val/test/unlabeled dataset')
    parser.add_argument('--output_path', type=str, default='logs',
                        help='Output directory for train or file path for infer/eval results')
    parser.add_argument('--recipe_hparams', nargs='+', action=DictAction,
                        help='Override default hyper-params in the recipe, the key-value pair in \
                              xxx=yyy format will be merged into recipe config.')
    parser.add_argument('--recipe_json',
                        help='Override/add hyper-params in the recipe, JSON configuration format will \
                              be translated as dict and merged into the recipe config.')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'infer', 'export'], default='train',
                        help='Switch the run mode')
    parser.add_argument('--stage_type', type=str, choices=get_available_types(), default=None,
                        help='if stage cfg is passed through recipe_cfg argument, you can designate its \
                              type using the argument')
    # parser.add_argument('--resume', type=bool, default=False, help='Switch to resume training from given checkpoint')

    # group_gpus = parser.add_mutually_exclusive_group()
    # group_gpus.add_argument(
    #     '--gpus',
    #     type=int,
    #     help='number of gpus to use '
    #     '(only applicable to non-distributed training)')
    # group_gpus.add_argument(
    #     '--gpu_ids',
    #     type=int,
    #     nargs='+',
    #     help='ids of gpus to use '
    #     '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--log_level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='set log level')

    main(parser.parse_args())
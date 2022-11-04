import argparse

from mmcv import DictAction

from mpa.utils.config_utils import MPAConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = MPAConfig.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    print(f'Config:\n{cfg.pretty_text}')


if __name__ == '__main__':
    main()

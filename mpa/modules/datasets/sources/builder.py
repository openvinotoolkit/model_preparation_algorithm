from mmcv.utils import Registry

# create a  registry for DATASOURCES
DATASOURCES = Registry('data_source')


def build_datasource(cfg, *args, **kwargs):
    cfg_ = cfg.copy()
    type_ = cfg_.pop('type')
    if type_ not in DATASOURCES:
        raise KeyError(f'Unrecognized task type {type_}')
    else:
        cls_ = DATASOURCES.get(type_)

    instance = cls_(*args, **kwargs, **cfg_)
    return instance

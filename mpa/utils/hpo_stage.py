import os
import time
import pickle
from multiprocessing import Process

from mmcv import build_from_cfg

from mpa.registry import STAGES
from mpa.stage import Stage
from mpa.utils.logger import get_logger

logger = get_logger()


def build_train_stage(trainer_stage_config, common_cfg):
    if trainer_stage_config is None:
        raise ValueError(
            'HpoRunner needs a trainer config to run.'
        )

    if trainer_stage_config.get('type', None) is None:
        raise ValueError(
            'The type of hpo trainer should be specified.'
        )
    if trainer_stage_config.get('config', None) is None:
        raise ValueError(
            'The file path of hpo trainer config should be specified.'
        )
    trainer_stage_config.name = 'hpo_trainer'
    trainer_stage_config.mode = 'train'
    trainer_stage_config.common_cfg = common_cfg

    return build_from_cfg(trainer_stage_config, STAGES)


def run_hpo_trainer(trainer_stage_config, common_cfg, metric, mode,
                    model_cfg, data_cfg, model_ckpt,
                    output_path, hp_config):
    trainer_stage = build_train_stage(trainer_stage_config, common_cfg)

    # replace old hyper-parameters with the newly created ones.
    if 'lr' in hp_config['params']:
        trainer_stage.cfg.optimizer.lr = hp_config['params']['lr']

    if 'bs' in hp_config['params']:
        data_cfg.data.samples_per_gpu = int(hp_config['params']['bs'])

    if hasattr(trainer_stage.cfg, 'runner'):
        trainer_stage.cfg.runner.max_epochs = hp_config['iterations']
    trainer_stage.cfg.total_epochs = hp_config['iterations']

    # add HPOHook
    hpo_hook = dict(type='HPOHook',
                    hp_config=hp_config,
                    metric=metric,
                    priority='LOW')
    custom_hooks = trainer_stage.cfg.get('custom_hooks', [])
    custom_hooks.append(hpo_hook)
    trainer_stage.cfg['custom_hooks'] = custom_hooks

    _ = trainer_stage.run(stage_idx=hp_config['trial_id'],
                          mode=mode,
                          model_cfg=model_cfg,
                          data_cfg=data_cfg,
                          model_ckpt=model_ckpt,
                          output_path=output_path,
                          hp_config=hp_config)


def exec_hpo_trainer(arg_file_name, gpu_id):
    trainer_file_name = os.path.join(os.path.dirname(__file__), 'hpo_trainer.py')
    os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH="{os.path.dirname(__file__)}/../../:$PYTHONPATH" '
              f'python {trainer_file_name} {arg_file_name}')
    time.sleep(10)
    return gpu_id


@STAGES.register_module()
class HpoRunner(Stage):
    def __init__(self, **kwargs):
        if kwargs.get('config', None) is None:
            kwargs['config'] = {'hpo': {'trainer': {'type': None, 'config': None}}}

        super().__init__(**kwargs)
        self.cfg['hpo'] = kwargs.pop('hpo', None)
        self.cfg['common_cfg'] = kwargs.pop('common_cfg', None)

        if self.cfg['hpo'] is None:
            raise ValueError(
                'Missing HPO configuration.'
            )

        self.hyperparams = self.cfg.hpo.get('hyperparams', None)
        if self.hyperparams is None:
            raise ValueError(
                'The list of hyper-parameters to tune is required to run HPO.'
            )

        self.metric = self.cfg.hpo.get('metric', None)
        if self.metric is None:
            raise ValueError(
                'The name of metric is required to run HPO.'
            )
        self.num_trials = self.cfg.hpo.get('num_trials', 10)
        self.max_iterations = self.cfg.hpo.get('max_iterations', 10)
        self.subset_size = self.cfg.hpo.get('subset_size', 0)
        self.search_alg = self.cfg.hpo.get('search_alg', 'bayes_opt')
        if self.search_alg not in ['bayes_opt', 'asha']:
            raise ValueError(
                'The \'search_alg\' should be one of \'bayes_opt\' or \'asha\'.'
            )
        self.gpu_list = self.cfg.hpo.get('gpu_list', [0])
        self.num_workers = len(self.gpu_list)
        self.num_brackets = self.cfg.hpo.get('num_brackets', 1)
        self.min_iterations = self.cfg.hpo.get('min_iterations', 1)
        self.reduction_factor = self.cfg.hpo.get('reduction_factor', 4)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        import hpopt

        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        output_path = self.cfg.work_dir

        hparams = {}
        for param in self.hyperparams:
            hparams[param.name] = hpopt.search_space(param.type, param.range)

        hpoptimizer = None

        if self.search_alg == 'bayes_opt':
            hpoptimizer = hpopt.create(save_path=output_path,
                                       search_alg='bayes_opt',
                                       search_space=hparams,
                                       ealry_stop='median_stop',
                                       num_init_trials=5,
                                       num_trials=self.num_trials,
                                       max_iterations=self.max_iterations,
                                       subset_size=self.subset_size)
        elif self.search_alg == 'asha':
            hpoptimizer = hpopt.create(save_path=output_path,
                                       search_alg='asha',
                                       search_space=hparams,
                                       num_trials=self.num_trials,
                                       min_iterations=self.min_iterations,
                                       max_iterations=self.max_iterations,
                                       reduction_factor=self.reduction_factor,
                                       num_brackets=self.num_brackets,
                                       subset_size=self.subset_size)

        logger.info('** HPO START **')

        if self.search_alg == 'bayes_opt':
            while True:
                hp_config = hpoptimizer.get_next_sample()

                if hp_config is None:
                    break

                run_hpo_trainer(trainer_stage_config=self.cfg.hpo.trainer.copy(),
                                common_cfg=self.cfg['common_cfg'].copy(),
                                metric=self.metric,
                                mode=mode,
                                model_cfg=model_cfg,
                                data_cfg=data_cfg,
                                model_ckpt=model_ckpt,
                                output_path=output_path,
                                hp_config=hp_config)
        elif self.search_alg == 'asha':
            proc_list = []
            gpu_alloc_list = []

            while True:
                num_active_workers = 0
                for p, g in zip(reversed(proc_list), reversed(gpu_alloc_list)):
                    if p.is_alive():
                        num_active_workers += 1
                    else:
                        p.close()
                        proc_list.remove(p)
                        gpu_alloc_list.remove(g)

                if num_active_workers == self.num_workers:
                    time.sleep(10)

                while num_active_workers < self.num_workers:
                    hp_config = hpoptimizer.get_next_sample()

                    if hp_config is None:
                        break

                    _kwargs = {"trainer_stage_config": self.cfg.hpo.trainer,
                               "common_cfg": self.cfg['common_cfg'],
                               "metric": self.metric,
                               "mode": mode,
                               "model_cfg": model_cfg,
                               "data_cfg": data_cfg,
                               "model_ckpt": model_ckpt,
                               "output_path": output_path,
                               "hp_config": hp_config}

                    pickle_path = os.path.join(output_path, f"hpo_trial_{hp_config['trial_id']}.pickle")
                    with open(pickle_path, "wb") as pfile:
                        pickle.dump(_kwargs, pfile)

                    gpu_id = -1
                    for idx in self.gpu_list:
                        if idx not in gpu_alloc_list:
                            gpu_id = idx
                            break

                    if gpu_id < 0:
                        raise ValueError('No available GPU!!')

                    p = Process(target=exec_hpo_trainer, args=(pickle_path, gpu_id, ))
                    proc_list.append(p)
                    gpu_alloc_list.append(gpu_id)
                    p.start()
                    num_active_workers += 1

                # All trials are done.
                if num_active_workers == 0:
                    break

        best_config = hpoptimizer.get_best_config()
        print("best hp: ", best_config)

        logger.info('** HPO END **')

        retval = {'hyperparams': {}}

        if 'lr' in best_config:
            retval['hyperparams']['lr'] = best_config.get('lr')

        if 'bs' in best_config:
            retval['hyperparams']['bs'] = best_config.get('bs')

        return retval

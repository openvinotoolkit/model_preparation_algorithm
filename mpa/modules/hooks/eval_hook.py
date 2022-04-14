# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from os import path as osp
import numpy as np
import mmcv
from mmcv.runner import Hook

import torch
from torch.utils.data import DataLoader


class CustomEvalHook(Hook):
    """Custom Evaluation hook for the MPA

    Args:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, by_epoch=True, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs
        self.by_epoch = by_epoch
        self.best_loss = 9999999.0
        self.best_score = 0.0
        self.save_mode = eval_kwargs.get('save_mode', 'score')
        metric = self.eval_kwargs['metric']

        self.metric = None
        if isinstance(metric, str):
            self.metric = 'top-1' if metric == 'accuracy' else metric
        else:
            self.metric = metric[0]
            if metric.count('class_accuracy') > 0:
                self.metric = 'accuracy'
            elif metric.count('accuracy') > 0:
                self.metric = 'top-1'

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        results = single_gpu_test(runner.model, self.dataloader)
        self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        score = self.cal_score(eval_res)
        if score >= self.best_score:
            self.best_score = score
            runner.save_ckpt = True

    def cal_score(self, res):
        score = 0
        div = 0
        for key, val in res.items():
            if np.isnan(val):
                continue
            if self.metric in key:
                score += val
                div += 1
        return score / div


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


class DistCustomEvalHook(CustomEvalHook):
    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 by_epoch=True,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.gpu_collect = gpu_collect
        super(DistCustomEvalHook, self).__init__(dataloader, interval, by_epoch=by_epoch, **eval_kwargs)

    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        from mmcls.apis import multi_gpu_test
        runner.log_buffer.clear()
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            self.evaluate(runner, results)

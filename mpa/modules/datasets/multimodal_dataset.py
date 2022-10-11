import os
from typing import List, Dict
from pathlib import Path
import numpy as np
import pandas as pd

from mmcls.datasets.base_dataset import BaseDataset
from torchvision.transforms import Compose
from sklearn.metrics import cohen_kappa_score

from mmcls.models.losses import accuracy
from mmcls.core.evaluation import precision_recall_f1, support

from mpa.multimodal.builder import DATASETS
from mpa.multimodal.builder import build_pipelines
from mpa.utils.logger import get_logger
from mpa.multimodal.preprocessor import TabularPreprocessor

SUPPORTED_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'support', \
                     'rmse', 'quadratic_kappa']

logger = get_logger()

@DATASETS.register_module()
class MultimodalDataset(BaseDataset):
    def __init__(self,
                 img_data=None, 
                 text_data=None, 
                 train_table_data=None,
                 val_table_data=None,
                 img_pipeline=None, 
                 text_pipeline=None, 
                 tabular_pipeline=None,
                 classes=None,
                 tabular_cfg=None,
                 img_col=None,
                 label_col=None,
                 is_train=None,
                 task_type=None,
                 sample_rate=None,
                 **kwargs):
        
        self.img_data = img_data
        self.text_data = text_data
        self.tabular_cfg = os.path.abspath('./samples/cfgs/data/'+tabular_cfg)
        self.img_col = img_col
        self.label_col = label_col
        self.is_train = is_train
        self.task_type = task_type

        self.img_data_list = self._get_image_files(self.img_data)
        self.table_df, self.labels = self._load_tabular_data(train_table_data=train_table_data,
                                                             val_table_data=val_table_data, 
                                                             cfg_path=self.tabular_cfg)
        self.img_to_idx = {self.table_df[self.img_col].iloc[i]: i for i in range(len(self.table_df))}
        self.table_df.drop(columns=[self.img_col, self.label_col], inplace=True)
        logger.info(f'Total {len(self.table_df.columns)} tabular data columns with :\n{list(self.table_df.columns)}')

        self.modalities = kwargs.get('modalities')
        self.data_infos = self.load_annotations(modalities=self.modalities)
        self.CLASSES = self.get_classes(classes)
                
        # build pipelines 
        if img_pipeline is not None:
            self.img_pipeline = Compose([build_pipelines(p) for p in img_pipeline])
        if text_pipeline is not None:
            self.text_pipeline = Compose([build_pipelines(p) for p in text_pipeline])
        if tabular_pipeline is not None: # TODO: need to finish implementation
            self.text_pipeline = Compose([build_pipelines(p) for p in tabular_pipeline])
            print('Pipeline for tabular is not implemented yet')
            raise
    
    def _get_image_files(self, root_dir):
        # recursively get all image file paths from given root_dir
        img_data_formats = ['.jpg', '.jpeg', '.JPEG', '.gif', '.bmp', '.tif', '.tiff', '.png']
        img_files = []
        for root, _, _ in os.walk(root_dir):
            for format in img_data_formats:
                img_files.extend([(root, file.name) for file in Path(root).glob(f'*{format}')])
        return img_files if img_files else None

    def _load_tabular_data(self, train_table_data, val_table_data, cfg_path=None):
        """
        Return pandas DataFrame with preprocessing.
        Both train and val table data should be given, because of one-hot encoding.
        """
        preprocessor = TabularPreprocessor(cfg_path)
        self.img_col = preprocessor.cfg['img_col']
        self.label_col = preprocessor.cfg['label_col']

        train_df, val_df = pd.read_csv(train_table_data), pd.read_csv(val_table_data)
        csv_data = train_df.append(val_df)
        csv_data = preprocessor.fill_na(csv_data)
        csv_data = preprocessor.normalize(csv_data)
        csv_data = preprocessor.categorical_feature_mapping(csv_data)
        labels = np.asarray(csv_data[self.label_col])
        meta_columns = preprocessor.get_meta_columns(csv_data)
        
        if self.is_train:
            return meta_columns[:len(train_df)], labels[:len(train_df)]
        else:
            return meta_columns[len(train_df):], labels[len(train_df):]

    def load_annotations(self, modalities: List[str]):
        """Load annotations by given modalities.

        Args:
            modalities : modalities of the datas

        Returns:
            out_data   : Containing infos about data
        """

        if 'vision' in modalities:
            logger.info(f'Total image datas   : {len(self.img_data_list)}')
        if 'tabular' in modalities:
            logger.info(f'Total tabular datas : {len(self.table_df)}')

        out_data = []
        for img_prefix, filename in self.img_data_list:
            img_key = filename.split('-')[0].split('.')[0]
            idx = self.img_to_idx[img_key]
            data_info = {'gt_label': int(self.labels[idx])}
            if 'vision' in modalities:
                data_info.update({
                    'img_prefix': img_prefix,
                    'img_info': {'filename': filename},
                    'meta_info': np.array(0, dtype=np.float32), ##TODO: consider other variable?
                })
            if 'tabular' in modalities:
                data_info.update({
                    'meta_info': np.array(self.table_df.iloc[idx], dtype=np.float32),
                })
            out_data.append(data_info)
        
        return out_data
            
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if self.task_type in ('classification', 'cls'):
            gt_label = np.asarray(data_info['gt_label']).astype('int64')
        elif self.task_type in ('regression', 'reg'):
            gt_label = np.asarray(data_info['gt_label']).astype('float32')
        results = {'meta_info': np.asarray(data_info['meta_info']).astype('float32'),
                   'gt_label': gt_label}
        if 'vision' in self.modalities:
            results.update(self.img_pipeline(data_info))
        return results
    
    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        assert len(gt_labels) == len(results), 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(SUPPORTED_METRICS)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        
        topk = metric_options.get('topk', (1, 5))
        average_mode = metric_options.get('average_mode', 'macro')
        if 'accuracy' in metrics:
            acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {f'accuracy_top-{k}': a for k, a in zip(topk, acc)}
            else:
                eval_results_ = {'accuracy': acc}
        
            eval_results.update({k: v.item() for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        if 'rmse' in metrics:
            rmse = np.sqrt(((results - np.expand_dims(gt_labels, axis=1))**2).mean())
            eval_results['rmse'] = rmse

        if 'quadratic_kappa' in metrics:
            preds = np.argmax(results, axis=-1)
            quadratic_kappa = cohen_kappa_score(preds, gt_labels, weights='quadratic')
            eval_results['quadratic_kappa'] = quadratic_kappa
        
        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    eval_results[key] = values

        return eval_results

import json

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from abc import ABC
import os

class BasePreprocessor(ABC):
    def __init__(self, cfg=None):
        self.cfg = self._load_json(cfg)
        self.meta_columns = list(self.cfg['categorical_features'].keys()) + \
                            list(self.cfg['numerical_features'].keys()) + \
                            [self.cfg['img_col']]

    def _load_json(self, cfg=None):
        if cfg is None:
            return dict()
        with open(cfg, 'r') as f:
            json_data = json.load(f)
        return json_data

class TabularPreprocessor(BasePreprocessor):
    def fill_na(self, df: pd.DataFrame):
        """ Fill N/A value. If not defined, categorical features will be automatically filled N/A
            values with -1, and numerical features will be filled the N/A values with mean value
            of the column.
        """
        for dtype in ['categorical_features', 'numerical_features']:
            for col, process in self.cfg[dtype].items():
                if 'fill_na' not in process:
                    if dtype == 'categorical_features':
                        df[col].fillna(-1, inplace=True)
                    elif dtype == 'numerical_features':
                        df[col].fillna(df[col].mean(), inplace=True)
                elif isinstance(process['fill_na'], (float, int)):
                    df[col].fillna(process['fill_na'], inplace=True)
                elif process['fill_na'] == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif process['fill_na'] == 'drop':
                    df.dropna(subset=[col], inplace=True)
        return df
    
    def normalize(self, df: pd.DataFrame):
        """ Normalize numerical values in each numerical data column.
        """
        for col, process in self.cfg['numerical_features'].items():
            if df[col].min() == df[col].max():
                continue
            elif process.get('norm', '') == 'minmax':
                min = process.get('min', 0)
                max = process.get('max', 1)
                scaler = MinMaxScaler(feature_range=(min, max))
                df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1))
            elif process['norm'] == 'gaussian':
                df[col] = (df[col]-df[col].mean()) / df[col].std()
        return df

    def categorical_feature_mapping(self, df: pd.DataFrame):
        """ Mapping data into encoded data. If column's encoding value is label_encoding,
            it encode and replace the value in column with int encoded value. If one_hot,
            then columns will be added for one-hot encoding.
        """
        for col, process in self.cfg['categorical_features'].items():
            if process['encoding'] == 'one_hot':
                prefix = process.get('prefix', col)
                dummies = pd.get_dummies(df[col], dummy_na=True, dtype=np.uint8, prefix=prefix)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=col, inplace=True)
                self.meta_columns.remove(col)
                self.meta_columns.extend(list(dummies.columns))
            elif process['encoding'] == 'label_encoding':
                encoding_dict = {data: i for i, data in enumerate(set(df[col]))}
                df[col] = df[col].apply(lambda x: encoding_dict[x])
        return df

    def get_meta_columns(self, df: pd.DataFrame):
        """ Get only columns that figured in cfg
        """
        return df[self.meta_columns]

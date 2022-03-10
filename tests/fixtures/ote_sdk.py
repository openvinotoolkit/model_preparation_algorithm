import os

import pytest
from mpa import MPAConstants
from ote_sdk.configuration.helper import create
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
    NullAnnotationSceneEntity,
)
from ote_sdk.entities.datasets import DatasetEntity, DatasetItemEntity, Subset
from ote_sdk.entities.image import Image
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters


@pytest.fixture(scope='function')
def fixture_task_env():
    model_template_path = os.path.join(
        MPAConstants.MODELS_PATH, 'templates/detection/MobilenetV2_ATSS_semi/template.yaml')
    model_template = parse_model_template(model_template_path)
    hyper_parameters = create(model_template.hyper_parameters.data)

    labels_schema = LabelSchemaEntity.from_labels([
        LabelEntity(name="0", domain=Domain.DETECTION),
        LabelEntity(name="1", domain=Domain.DETECTION),
        LabelEntity(name="2", domain=Domain.DETECTION),
    ])

    task_env = TaskEnvironment(
        model=None,
        hyper_parameters=hyper_parameters,
        label_schema=labels_schema,
        model_template=model_template
    )
    return task_env


# @pytest.fixture(scope='function')
# def fixture_det_hparams():
#     hyper_parameters = Mock(spec=DetectionConfig)
#     return hyper_parameters


def create_dataset_item_entity(target_path, subset):
    items = []
    for dirpath, dirs, files in os.walk(target_path):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            label = str(dirpath.split('/')[-1])
            label = LabelEntity(id=-1, name=label, domain=Domain.DETECTION)
            if fname.endswith('.png'):
                item = DatasetItemEntity(
                    media=Image(file_path=fname),
                    annotation_scene=NullAnnotationSceneEntity() if subset == Subset.UNLABELED
                    else AnnotationSceneEntity(
                        annotations=[
                            Annotation(Rectangle(x1=0, y1=0, x2=1, y2=1), labels=[ScoredLabel(label)])
                        ],
                        kind=AnnotationSceneKind.ANNOTATION
                    ),
                    subset=subset
                )
                items.append(item)
    return items


@pytest.fixture(scope='function')
def fixture_dataset_entity():
    total_items = []

    target_path = os.path.join(MPAConstants.TESTS_PATH, 'assets/dirs/classification/train')
    total_items.extend(create_dataset_item_entity(target_path, Subset.TRAINING))

    target_path = os.path.join(MPAConstants.TESTS_PATH, 'assets/dirs/classification/val')
    total_items.extend(create_dataset_item_entity(target_path, Subset.VALIDATION))

    target_path = os.path.join(MPAConstants.TESTS_PATH, 'assets/dirs/classification/unlabeled')
    total_items.extend(create_dataset_item_entity(target_path, Subset.UNLABELED))

    return DatasetEntity(items=total_items)


# @pytest.fixture(scope='function')
# def fixture_model_entity():
#     model_entity = Mock(spec=ModelEntity)
#     return model_entity


@pytest.fixture(scope='function')
def fixture_train_parameters():
    train_parameters = TrainParameters()
    return train_parameters

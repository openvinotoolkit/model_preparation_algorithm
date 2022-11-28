"""Utils for detection task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mpa.det.incr_stage import IncrDetectionStage


def load_patcher(training_type, **kwargs):
    if training_type == 'incremental':
        patcher = IncrDetectionStage(**kwargs)
    else:
        raise NotImplementedError(f"{training_type} is not supported in detection task")
    return patcher

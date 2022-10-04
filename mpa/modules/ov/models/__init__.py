# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa
try:
    import mmcls as mmcls_import_test
except ImportError:
    pass
else:
    from . import mmcls

try:
    import mmdet as mmdet_import_test
except ImportError:
    pass
else:
    from . import mmdet

try:
    import mmseg as mmseg_import_test
except ImportError:
    pass
else:
    from . import mmseg

from .ov_model import OVModel

from mpa.modules.xai.explain_algorithms import RISE

from mpa.registry import EXPLAINERS

def build_explainer(model):
    return RISE(model)
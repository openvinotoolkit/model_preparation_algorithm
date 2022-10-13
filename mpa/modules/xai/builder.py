from mpa.modules.xai.explain_algorithms import RISE


def build_explainer(model, cfg):
    return RISE(model, cfg)
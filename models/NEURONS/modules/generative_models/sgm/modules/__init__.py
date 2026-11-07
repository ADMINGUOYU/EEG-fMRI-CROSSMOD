from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "models.NEURONS.modules.generative_models.sgm.modules.GeneralConditioner",
    "params": {"emb_models": []},
}

from omegaconf import OmegaConf
import logging


def custom_eval(x):
    logging.debug(x)
    return eval(x)


if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", custom_eval)

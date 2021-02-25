import os
import torch
import random
import numpy as np
from logging import Formatter, StreamHandler, getLogger


def get_logger():
    log_fmt = Formatter(
        "%(asctime)s [%(levelname)s][%(funcName)s] %(message)s "
    )
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(log_fmt)
    logger.setLevel("INFO")
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def seed_everything(seed=1006):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

""" Base Params
"""
SEED = 1006
N_CLASS = 5


""" EXP Params
"""
N_SPLITS = 5
KFOLD = "StratifiedKFold"


""" Training Params
"""
LR = 1e-4
EPOCH = 30
MAX_PATIENCE = 3
BATCH_SIZE = 32
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


""" PATH (※ PATH depends on your env
"""
PATH = "/home/cassava"
DATA_PATH = f"{PATH}/data"
OUTPUT_PATH = f"{PATH}/output"

# ~ - data - train - ***.jpg
#                  |       - ***.jpg ...
#                  - merged.csv
#          - output

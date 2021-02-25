import yaml
import pprint
import argparse
from attrdict import AttrDict
from src.runner import Runner
from src import settings


def main(is_debug):
    """ Training Pipeline
    """
    with open("./config.yaml") as yf:
        config = yaml.safe_load(yf)

    # run single models
    for config_ in config["models"]:
        pprint.pprint(config_)
        runner = Runner(settings, AttrDict(config_))
        runner.run(is_debug=args.debug, multi_gpu=args.multi_gpu)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run pipeline")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--multi-gpu", action="store_true", default=False)
    args = parser.parse_args()

    main(args)

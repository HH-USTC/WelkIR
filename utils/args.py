from argparse import ArgumentParser
from loguru import logger

def add_args(parser: ArgumentParser):
    # data and saving
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="input yaml file ")


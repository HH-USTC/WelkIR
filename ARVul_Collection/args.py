from argparse import ArgumentParser
from loguru import logger

def add_args(parser: ArgumentParser):
    # data and saving
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="yaml file ")


def check_args(args):
    """Check if args values are valid, and conduct some default settings."""
    logger.info(f" check_args ")
    
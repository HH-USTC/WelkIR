from transformers import (RobertaTokenizer, RobertaModel, DataCollatorForLanguageModeling, AdamW,
                          get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader, RandomSampler
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import os
from loguru import logger

from utils.args import add_args
import yaml
from models.vuldetect.run_vuldetect_task import run_vuldetect_task
from models.pretrain_model.run_pretrain_module import run_pretrain_module, generate_pretrain_PKfile

from models.vulclassify.run_vulclassify_task import run_vulclassify_task



logger.add("WelkIR_Module_info.log", rotation="10 MB", encoding="utf-8", enqueue=True)

class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7,8,9"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ["yes", "true", "t", "1", "y"])

    add_args(parser)
    args = parser.parse_args()
    # check args
    logger.info(f" args.config: {args.config} ")
    config_dict  = load_config(args.config)
    config = Config(config_dict)

    if config.task == "pretrain":
        pass

    elif config.task == "vuldetect":
        logger.info(os.getcwd())
        config_dict  = load_config("models/vuldetect/vuldetect_config.yaml")
        config = Config(config_dict)
        run_vuldetect_task(config)
    elif config.task == "vulclassify":
        logger.info(f" start vulclassify ")
        config_dict  = load_config("models/vulclassify/vulclassify_config.yaml")
        config = Config(config_dict)
        run_vulclassify_task(config)
    
        
        

if __name__ == "__main__":
    logger.info(f" start Welkir ... ")
    main()




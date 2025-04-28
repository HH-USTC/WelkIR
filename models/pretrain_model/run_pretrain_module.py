from loguru import logger
from tqdm import tqdm
import torch
import math
import random
import os
import sys
from utils.utils import (
    load_tokenizer, prepare_pretrain_dataset, prepare_pretrain_dataset_to_pkfile, 
    MixedPrecisionManager, EarlyStopController, load_pickle, process_pkfile_to_List,
    Mul_PretrainIterableDataset
)
from torch.utils.data import Dataset
from torch.nn.parallel import DataParallel
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, List

from .configuration_welkir import WelkirConfig
from .pretrain_module_welk  import WelkirPreTrainedLearningModel
from transformers import (get_scheduler, DataCollatorForLanguageModeling)
from transformers.modeling_utils import PreTrainedModel




# class Mul_PretrainDataset(Dataset):    
#     def __init__(self, data_paths, batch_size, shuffle_files=True):
#         self.data_paths = data_paths  # List of all pickle files
#         self.batch_size = batch_size
#         self.file_sizes = [self._get_file_size(path) for path in data_paths]  # Calculate file sizes
#         self.cumulative_sizes = self._get_cumulative_sizes(self.file_sizes)  # Calculate cumulative sizes
#         self.current_file_index = 0  # Index of the current file
#         self.current_data_index = 0  # Index of the current data entry
#         self.dataset = None  # Lazy loading of data
#         self.shuffle_files = shuffle_files
#         if self.shuffle_files:
#             random.shuffle(self.data_paths)  # 如果需要，打乱文件的顺序
#     def _get_file_size(self, path):
#         """Calculate the number of entries in a pickle file."""
#         if not os.path.exists(path):
#             raise ValueError(f"Pickle file not exists: {os.path.abspath(path)}")
#         data = load_pickle(path)  
#         return len(data)
#     def _get_cumulative_sizes(self, file_sizes):
#         """Calculate the cumulative sizes of all files."""
#         cumulative_sizes = []
#         total = 0
#         for size in file_sizes:
#             total += size
#             cumulative_sizes.append(total)
#         return cumulative_sizes
#     def load_data(self, path):
#         """Load data from a pickle file."""
#         if not os.path.exists(path):
#             raise ValueError(f"Pickle file not exists: {os.path.abspath(path)}")
#         data = load_pickle(path)  # Assuming `load_pickle` is a wrapper around `pickle.load`
#         return data
#     def __len__(self):
#         return self.cumulative_sizes[-1]
#     # Retrieve data based on index
#     def __getitem__(self, idx):
#         """Retrieve a single data item."""
#         # Determine which file the idx belongs to
#         file_idx = next(i for i, cumulative_size in enumerate(self.cumulative_sizes) if cumulative_size > idx)
#         # Calculate the index within the file
#         local_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
#         # If we are accessing a new file, load it
#         if self.dataset is None or file_idx != self.current_file_index:
#             self.dataset = self.load_data(self.data_paths[file_idx])
#             self.current_file_index = file_idx
#             self.current_data_index = 0
#         # Return the feature at the local index in the current file
#         feature = self.dataset[local_idx]
#         return feature
    

def save_model(model, optimizer, lr_scheduler, scaler, config, save_dir, epoch):
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),              
        "optimizer_state_dict": optimizer.state_dict(),              
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),      
        "epoch": epoch,                                         
        "config": config,                                
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    save_path = os.path.join(save_dir, "pretrain_model_checkpoint.pt")
    torch.save(checkpoint, save_path)
    logger.info(f"Model, optimizer, and scheduler checkpoint for epoch {epoch} saved to {save_path}")

def load_checkpoint_if_exists(pretrain_module, optimizer, lr_scheduler, scaler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        state_dict = checkpoint["model_state_dict"]
        if isinstance(pretrain_module, torch.nn.DataParallel):
            if not list(state_dict.keys())[0].startswith('module.'):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[f'module.{k}'] = v  # 添加 'module.' 前缀
                state_dict = new_state_dict
        pretrain_module.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        logger.info(f"Resuming from epoch {start_epoch}.")
        return start_epoch
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        return 0

def generate_pretrain_PKfile(config):
    logger.info(f" start generate_pretrain_PKfile ")
    tokenizer, cfg_vocab, dfg_vocab, rdg_vocab = load_tokenizer(config)
    split = config.dataset_split
    prepare_pretrain_dataset_to_pkfile(
        config=config,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab,
        split=split,
    )

# get pickle file
def get_data_paths(dataset_root, file_suffix=".pk"):
    cache_dir = os.path.join(dataset_root, 'cache')
    if not os.path.exists(cache_dir):
        raise ValueError(f"Cache directory does not exist: {cache_dir}")
    
    data_paths = []
    total_samples = 0
    for file_name in sorted(os.listdir(cache_dir)):
        if file_name.startswith("pretrain-Vul_Func-dataset-") and file_name.endswith(f"{file_suffix}"):
            file_path = os.path.join(cache_dir, file_name)
            data_paths.append(file_path)
        elif file_name.startswith("pretrain-Juliet-") and file_name.endswith(f"{file_suffix}"):
            logger.info(f" file_name:{file_name}")
            file_path = os.path.join(cache_dir, file_name)
            features = load_pickle(file_path) 
            total_samples += len(features)
            if not isinstance(features, list):
                logger.error(f"The file {file_path} does not contain a list of features.")
                continue
            data_paths.append(file_path)
            logger.info(f"file_path:{file_path} , total_samples:{total_samples}")
        elif file_name.startswith("pretrain-ARVO-") and file_name.endswith(f"{file_suffix}"):
            logger.info(f" file_name:{file_name}")
            file_path = os.path.join(cache_dir, file_name)
            features = load_pickle(file_path) 
            total_samples += len(features)
            if not isinstance(features, list):
                logger.error(f"The file {file_path} does not contain a list of features.")
                continue
            data_paths.append(file_path)
            logger.info(f"file_path:{file_path} , total_samples:{total_samples}")
        else:
            logger.info(f" file_name:{file_name}")
            file_path = os.path.join(cache_dir, file_name)
            features = load_pickle(file_path) 
            total_samples += len(features)
            if not isinstance(features, list):
                logger.error(f"The file {file_path} does not contain a list of features.")
                continue
            data_paths.append(file_path)
            logger.info(f"file_path:{file_path} , total_samples:{total_samples}")


    if not data_paths:
        raise ValueError(f"No pickle files found in {cache_dir} matching the pattern.")
    
    return data_paths, total_samples


def create_dataloader(config, dataset_root, shuffle=False, pin_memory=True):
   
    # pk_file_paths = get_data_paths(config.dataset_root, ".pk")
    # Listfiles_paths, total_samples = process_pkfile_to_List(pk_file_paths)
    
    Listfiles_paths, total_samples  = get_data_paths(dataset_root, ".pk")
    
    num_workers = config.DataLoader_num_workers
    logger.info(f"Listfiles Count: {len(Listfiles_paths)}, total_samples: {total_samples}")

    dataset = Mul_PretrainIterableDataset(Listfiles_paths)
 
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
       
    return dataloader, total_samples


def run_pretrain_module(config):
    pass
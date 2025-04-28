from loguru import logger
import os
import pickle
import re
import torch.nn as nn
import random
import numpy as np
import multiprocessing
import json
import sys
from tqdm import tqdm
import torch
import gc
from sklearn.utils import shuffle
from dataclasses import dataclass
from typing import Tuple, List, Optional
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset, DataLoader
from functools import partial
from prettytable import PrettyTable
from transformers import (
    PreTrainedTokenizerFast,
    RobertaTokenizerFast,
    AutoModel,
    AutoConfig,
)


def load_tokenizer(config):
    logger.info("Start loading tokenizer and vocabs...")

    tokenizer = RobertaTokenizerFast.from_pretrained(
        os.path.join(config.vocab_root, config.tokenizer_dir)
    )
    # tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    config.vocab_size = tokenizer.vocab_size

    config.bos_token_id = tokenizer.bos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.mask_token_id = tokenizer.mask_token_id

    config.bos_token = tokenizer.bos_token
    config.pad_token = tokenizer.pad_token
    config.eos_token = tokenizer.eos_token
    config.mask_token = tokenizer.mask_token

    logger.info(f"Tokenizer loaded, size: {config.vocab_size}")

    with open(os.path.join(config.vocab_root, config.cfg_vocab_name), mode="r", encoding="utf-8") as f:
        cfg_vocab = json.load(f)
    config.cfg_vocab_size = len(cfg_vocab)
    logger.info(f"CFG flow type vocabulary loaded, size: {config.cfg_vocab_size}")

    with open(os.path.join(config.vocab_root, config.dfg_vocab_name), mode="r", encoding="utf-8") as f:
        dfg_vocab = json.load(f)
    config.dfg_vocab_size = len(dfg_vocab)
    logger.info(f"DFG flow type vocabulary loaded, size: {config.dfg_vocab_size}")

    with open(os.path.join(config.vocab_root, config.rdg_vocab_name), mode="r", encoding="utf-8") as f:
        rdg_vocab = json.load(f)
    config.rdg_vocab_size = len(rdg_vocab)
    logger.info(f"RDG flow type vocabulary loaded, size: {config.rdg_vocab_size}")

    return tokenizer, cfg_vocab, dfg_vocab, rdg_vocab


@dataclass
class Example(object):
    base_filename: str
    func_slice_num: int
    IRcode: list
    src_nodeid: int
    dst_nodeid: int
    Linenum: int
    label_idx: int

# all_inst_encoder_input_ids: 各个指令分组的token编码。
# input_instcount_list: 每个分组的指令数量。
# input_inst_tokencount: 每个指令的token数量。
# input_ids: 用于 WelkirModel 的指令掩码输入。
# cfg_matrix: 控制流图矩阵。
# dfg_matrix: 数据流图矩阵。
# rdg_matrix: 可达性依赖图矩阵。


@dataclass
class InputFeature(object):
    input_ids: torch.Tensor 
    all_inst_encoder_input_ids: torch.Tensor  
    input_instcount_list: torch.Tensor 
    input_inst_tokencount: torch.Tensor
    cfg_matrix: torch.Tensor  
    dfg_matrix: torch.Tensor  
    rdg_matrix: torch.Tensor 
    cfg_matrix_mask: torch.Tensor
    dfg_matrix_mask: torch.Tensor
    rdg_matrix_mask: torch.Tensor
    total_group_count: torch.Tensor
    labels: torch.Tensor
    mask_all_inst_encoder_input_ids: torch.Tensor
    inst_mlm_labels: torch.Tensor

 
@dataclass
class VulDetect_InputFeature(object):
    input_ids: torch.Tensor 
    all_inst_encoder_input_ids: torch.Tensor  
    input_instcount_list: torch.Tensor 
    input_inst_tokencount: torch.Tensor
    cfg_matrix: torch.Tensor  
    dfg_matrix: torch.Tensor  
    rdg_matrix: torch.Tensor 
    total_group_count: torch.Tensor
    labels: torch.Tensor
    inst_mlm_labels: torch.Tensor
    base_filename: torch.Tensor 
    # add 20250323 attention
    func_slice_num: torch.Tensor 
    src_nodeid: torch.Tensor 
    dst_nodeid: torch.Tensor 

class PretrainDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
    #   return self.features[item].__dict__
        feature = self.features[item]
        # Mask the instruction sequence of the current feature
        masked_input_ids, inst_mlm_labels, attention_mask = self.mask_inst_tokens(feature.all_inst_encoder_input_ids)
        cfg_matrix_mask = self.mask_matrix_relationship(feature.cfg_matrix)
        dfg_matrix_mask = self.mask_matrix_relationship(feature.dfg_matrix)
        rdg_matrix_mask = self.mask_matrix_relationship(feature.rdg_matrix)
        
        return {
            'input_ids': feature.input_ids,
            'all_inst_encoder_input_ids': feature.all_inst_encoder_input_ids,
            'input_instcount_list': feature.input_instcount_list,
            'input_inst_tokencount': feature.input_inst_tokencount,
            'cfg_matrix': feature.cfg_matrix,
            'dfg_matrix': feature.dfg_matrix,
            'rdg_matrix': feature.rdg_matrix,
            'total_group_count': feature.total_group_count,
            'labels': feature.labels,
            'mask_all_inst_encoder_input_ids': masked_input_ids,
            'inst_mlm_labels': inst_mlm_labels,
            'attention_mask': attention_mask,
            'cfg_matrix_mask': cfg_matrix_mask,
            'dfg_matrix_mask': dfg_matrix_mask,
            'rdg_matrix_mask': rdg_matrix_mask,
        }

    def __len__(self):
        return len(self.features)
    

    # Predict the relationship data selected by mask_matrix_relationship
    def mask_matrix_relationship(self, matrix, mlm_probability=0.10):
        # matrix.shape: (num_rows, num_cols)
        matrix_mask = torch.zeros_like(matrix, dtype=torch.bool)
        
        no_relation_indices = torch.where(matrix == 1)
        no_relation_length = no_relation_indices[0].size(0)
        has_relation_indices = torch.where(matrix > 1)
        has_relation_length = has_relation_indices[0].size(0)
        
        total_mask_count = int(matrix.numel() * mlm_probability)
        n_mask_has_relation = min(has_relation_length, total_mask_count // 2)
        n_mask_no_relation = total_mask_count - n_mask_has_relation
        

        if has_relation_length < total_mask_count // 2:
            n_mask_no_relation = min(no_relation_length, total_mask_count - has_relation_length)
       
        if no_relation_length < total_mask_count // 2:
            n_mask_has_relation = min(has_relation_length, total_mask_count - no_relation_length)
        
        # Randomly select indices of unrelated positions
        if n_mask_no_relation > 0:
            permuted_indices_no_relation = torch.randperm(no_relation_length)[:n_mask_no_relation]
            selected_no_relation_row_indices = no_relation_indices[0][permuted_indices_no_relation]
            selected_no_relation_col_indices = no_relation_indices[1][permuted_indices_no_relation]
            matrix_mask[selected_no_relation_row_indices, selected_no_relation_col_indices] = True

        # Randomly select indices of related positions
        if n_mask_has_relation > 0:
            permuted_indices_has_relation = torch.randperm(has_relation_length)[:n_mask_has_relation]
            selected_has_relation_row_indices = has_relation_indices[0][permuted_indices_has_relation]
            selected_has_relation_col_indices = has_relation_indices[1][permuted_indices_has_relation]
            matrix_mask[selected_has_relation_row_indices, selected_has_relation_col_indices] = True

        return matrix_mask

    # Mask the input tokens for MLM tasks
    def mask_inst_tokens(self, input_ids, mlm_probability=0.15):
        num_groups, seq_length = input_ids.shape
        input_ids = input_ids.reshape(num_groups, seq_length)
        labels = input_ids.clone()
        # Compute the number of valid tokens, filtering out 1 and 0, as they are treated as padding or special tokens.
        valid_lengths = ((input_ids != 1) & (input_ids != 0)).sum(dim=1)
        # logger.info(f"valid_lengths shape: {valid_lengths.shape}, valid_lengths:{valid_lengths}")
        # Create a mask tensor to indicate which tokens are masked
        mask = torch.zeros_like(input_ids, dtype=torch.bool)      
        for i in range(input_ids.size(0)):  
            valid_length = valid_lengths[i].item()
            if valid_length == 0:
                continue
            n_mask = int(valid_length * mlm_probability) + 1
            valid_indices = torch.where((input_ids[i] != 1) & (input_ids[i] != 0))[0]   
            masked_indices = valid_indices[torch.randperm(valid_length)[:n_mask]]
            mask[i, masked_indices] = True

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = 4      #tokenizer.mask_token_id = 4 
        labels[~mask] = -100
        attention_mask = ((input_ids != 1) & (input_ids != 0)).long()
        masked_input_ids = masked_input_ids.reshape(num_groups, seq_length)
        labels = labels.reshape(num_groups, seq_length)
        attention_mask = attention_mask.reshape(num_groups, seq_length)
        return masked_input_ids, labels, attention_mask






# vul_detect
class Mul_Vul_IterableDataset(IterableDataset):
    def __init__(
        self,
        list_file_paths: List[str],
        shuffle_files: bool = True,
    ):
        super(Mul_Vul_IterableDataset, self).__init__()
        self.list_file_paths = list_file_paths.copy()
        self.shuffle_files = shuffle_files

        if self.shuffle_files:
            random.shuffle(self.list_file_paths)

    def load_data(self, path: str):
        if not os.path.exists(path):
            raise ValueError(f".list file does not exist: {os.path.abspath(path)}")
        data = self.load_pickle(path)
        if not isinstance(data, list):
            raise ValueError(f"Data in {path} is not a list.")
        return data

    @staticmethod
    def load_pickle(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    # Load data file by file and iterate through samples
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_files = self.list_file_paths[worker_id::num_workers]
        logger.info(f"Worker {worker_id} processing {len(worker_files)} .list files.")

        for file_path in worker_files:
            try:
                data = self.load_data(file_path)
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
                continue  

            for sample in data:
                processed_sample = self.process_sample(sample)
                yield processed_sample

            # Release memory
            del data
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug(f"Released memory for file {file_path}.")

    def process_sample(self, sample):
        processed_sample = {
                'input_ids': sample.input_ids,
                'all_inst_encoder_input_ids': sample.all_inst_encoder_input_ids,
                'input_instcount_list': sample.input_instcount_list,
                'input_inst_tokencount': sample.input_inst_tokencount,
                'cfg_matrix': sample.cfg_matrix,
                'dfg_matrix': sample.dfg_matrix,
                'rdg_matrix': sample.rdg_matrix,
                'total_group_count': sample.total_group_count,
                'labels': sample.labels,
                'base_filename': sample.base_filename,
                # add attention
                # 
                'func_slice_num': sample.func_slice_num,
                'src_nodeid': sample.src_nodeid,
                'dst_nodeid': sample.dst_nodeid,
            }
        return processed_sample
    


# class detect
class class_Mul_Vul_IterableDataset(IterableDataset):
    def __init__(
        self,
        list_file_paths: List[str],
        shuffle_files: bool = True,
    ):
        super(class_Mul_Vul_IterableDataset, self).__init__()
        self.list_file_paths = list_file_paths.copy()
        self.shuffle_files = shuffle_files

        if self.shuffle_files:
            random.shuffle(self.list_file_paths)

    def load_data(self, path: str):
        if not os.path.exists(path):
            raise ValueError(f".list file does not exist: {os.path.abspath(path)}")
        data = self.load_pickle(path)
        if not isinstance(data, list):
            raise ValueError(f"Data in {path} is not a list.")
        return data

    @staticmethod
    def load_pickle(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    # Load data file by file and iterate through samples
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_files = self.list_file_paths[worker_id::num_workers]
        logger.info(f"Worker {worker_id} processing {len(worker_files)} .list files.")

        for file_path in worker_files:
            try:
                data = self.load_data(file_path)
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
                continue  

            for sample in data:
                processed_sample = self.process_sample(sample)
                yield processed_sample

            # Release memory
            del data
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug(f"Released memory for file {file_path}.")

    def process_sample(self, sample):
        processed_sample = {
                'input_ids': sample.input_ids,
                'all_inst_encoder_input_ids': sample.all_inst_encoder_input_ids,
                'input_instcount_list': sample.input_instcount_list,
                'input_inst_tokencount': sample.input_inst_tokencount,
                'cfg_matrix': sample.cfg_matrix,
                'dfg_matrix': sample.dfg_matrix,
                'rdg_matrix': sample.rdg_matrix,
                'total_group_count': sample.total_group_count,
                'labels': sample.labels,
                'base_filename': sample.base_filename,
            }
        return processed_sample
    
    

class Mul_PretrainIterableDataset(IterableDataset):
    def __init__(
        self,
        list_file_paths: List[str],
        shuffle_files: bool = True,
    ):
        super(Mul_PretrainIterableDataset, self).__init__()
        self.list_file_paths = list_file_paths.copy()
        self.shuffle_files = shuffle_files

        if self.shuffle_files:
            random.shuffle(self.list_file_paths)

    def load_data(self, path: str):
        if not os.path.exists(path):
            raise ValueError(f".list file does not exist: {os.path.abspath(path)}")
        data = self.load_pickle(path)
        if not isinstance(data, list):
            raise ValueError(f"Data in {path} is not a list.")
        return data

    @staticmethod
    def load_pickle(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    # Load data file by file and iterate through samples
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_files = self.list_file_paths[worker_id::num_workers]
        logger.info(f"Worker {worker_id} processing {len(worker_files)} .list files.")

        for file_path in worker_files:
            try:
                data = self.load_data(file_path)
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
                continue  

            for sample in data:
                processed_sample = self.process_sample(sample)
                yield processed_sample

            # Release memory
            del data
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug(f"Released memory for file {file_path}.")

    def process_sample(self, sample):

        all_inst_encoder_input_ids = sample.all_inst_encoder_input_ids
        cfg_matrix = sample.cfg_matrix
        dfg_matrix = sample.dfg_matrix
        rdg_matrix = sample.rdg_matrix
        
        masked_input_ids, inst_mlm_labels, attention_mask = self.mask_inst_tokens(all_inst_encoder_input_ids)
        cfg_matrix_mask = self.mask_matrix_relationship(cfg_matrix)
        dfg_matrix_mask = self.mask_matrix_relationship(dfg_matrix)
        rdg_matrix_mask = self.mask_matrix_relationship(rdg_matrix)

        processed_sample = {
                'input_ids': sample.input_ids,
                'all_inst_encoder_input_ids': sample.all_inst_encoder_input_ids,
                'input_instcount_list': sample.input_instcount_list,
                'input_inst_tokencount': sample.input_inst_tokencount,
                'cfg_matrix': sample.cfg_matrix,
                'dfg_matrix': sample.dfg_matrix,
                'rdg_matrix': sample.rdg_matrix,
                'total_group_count': sample.total_group_count,
                'labels': sample.labels,
                'mask_all_inst_encoder_input_ids': masked_input_ids,
                'inst_mlm_labels': inst_mlm_labels,
                'attention_mask': attention_mask,
                'cfg_matrix_mask': cfg_matrix_mask,
                'dfg_matrix_mask': dfg_matrix_mask,
                'rdg_matrix_mask': rdg_matrix_mask,
            }

        return processed_sample
    
    # Predict the data for positions selected by mask_matrix_relationship
    def mask_matrix_relationship(self, matrix, mlm_probability=0.10):
        # matrix.shape: (num_rows, num_cols)

        matrix_mask = torch.zeros_like(matrix, dtype=torch.bool)
        no_relation_indices = torch.where(matrix == 1)
        no_relation_length = no_relation_indices[0].size(0)
        has_relation_indices = torch.where(matrix > 1)
        has_relation_length = has_relation_indices[0].size(0)
        
        total_mask_count = int(matrix.numel() * mlm_probability)
        n_mask_has_relation = min(has_relation_length, total_mask_count // 2)
        n_mask_no_relation = total_mask_count - n_mask_has_relation
        if has_relation_length < total_mask_count // 2:
            n_mask_no_relation = min(no_relation_length, total_mask_count - has_relation_length)
       
        if no_relation_length < total_mask_count // 2:
            n_mask_has_relation = min(has_relation_length, total_mask_count - no_relation_length)
        if n_mask_no_relation > 0:
            permuted_indices_no_relation = torch.randperm(no_relation_length)[:n_mask_no_relation]
            selected_no_relation_row_indices = no_relation_indices[0][permuted_indices_no_relation]
            selected_no_relation_col_indices = no_relation_indices[1][permuted_indices_no_relation]
            matrix_mask[selected_no_relation_row_indices, selected_no_relation_col_indices] = True
        if n_mask_has_relation > 0:
            permuted_indices_has_relation = torch.randperm(has_relation_length)[:n_mask_has_relation]
            selected_has_relation_row_indices = has_relation_indices[0][permuted_indices_has_relation]
            selected_has_relation_col_indices = has_relation_indices[1][permuted_indices_has_relation]
            matrix_mask[selected_has_relation_row_indices, selected_has_relation_col_indices] = True
        return matrix_mask
    
    # Mask the input tokens for MLM tasks
    def mask_inst_tokens(self, input_ids, mlm_probability=0.15, mask_token_id=4, random_token_range=(5, 29999)):
        # logger.info(f"PretrainDataset input_ids.shape:{input_ids.shape}")

        num_groups, seq_length = input_ids.shape
        input_ids = input_ids.reshape(num_groups, seq_length)
        labels = input_ids.clone()
        valid_lengths = ((input_ids != 1) & (input_ids != 0)).sum(dim=1)
        # logger.info(f"valid_lengths shape: {valid_lengths.shape}, valid_lengths:{valid_lengths}")
        mask = torch.zeros_like(input_ids, dtype=torch.bool)      

        masked_input_ids = input_ids.clone()
        for i in range(input_ids.size(0)):  
            valid_length = valid_lengths[i].item()  
            if valid_length == 0:
                continue
            n_mask = int(valid_length * mlm_probability) + 1
            valid_indices = torch.where((input_ids[i] != 1) & (input_ids[i] != 0))[0]   
            masked_indices = valid_indices[torch.randperm(valid_length)[:n_mask]]
            mask[i, masked_indices] = True

            # Obtain the indices to be replaced with random tokens (10%)
            random_indices = masked_indices.clone()
            num_random = int(len(masked_indices) * 0.1)
            random_masked_indices = random_indices[torch.randperm(len(random_indices))[:num_random]]
            remaining_indices = masked_indices[~torch.isin(masked_indices, random_masked_indices)]
            unchanged_indices = remaining_indices.clone()
            num_unchanged = int(len(masked_indices) * 0.1)
            unchanged_masked_indices = unchanged_indices[torch.randperm(len(unchanged_indices))[:num_unchanged]]
            random_tokens = torch.randint(random_token_range[0], random_token_range[1], (len(random_masked_indices),), dtype=masked_input_ids.dtype, device=masked_input_ids.device)
            masked_input_ids[i, masked_indices] = mask_token_id
            masked_input_ids[i, random_masked_indices] = random_tokens
            masked_input_ids[i, unchanged_masked_indices] = input_ids[i, unchanged_masked_indices]
            
        labels[~mask] = -100
        attention_mask = ((input_ids != 1) & (input_ids != 0)).long()
        masked_input_ids = masked_input_ids.reshape(num_groups, seq_length)
        labels = labels.reshape(num_groups, seq_length)
        attention_mask = attention_mask.reshape(num_groups, seq_length)
        return masked_input_ids, labels, attention_mask



def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    if not os.path.exists(path):
        raise ValueError(f"Pickle file not exists: {os.path.abspath(path)}")
    with open(path, mode="rb") as f:
        obj = pickle.load(f)
    return obj


# Convert pk files to list files.
def process_pkfile_to_List(pk_files):
    list_file_paths = []
    total_samples = 0
    if not pk_files:
        logger.warning("No .pk files provided for processing.")
        return list_file_paths
    
    logger.info(f"Found pk files : {pk_files}")
    for pk_file in pk_files:
        try:
            pretrain_dataset = load_pickle(pk_file) 
            if not isinstance(pretrain_dataset, PretrainDataset):
                logger.error(f"The file {pk_file} does not contain a PretrainDataset object.")
                continue       
            # 提取 features 列表
            features = pretrain_dataset.features
            if not isinstance(features, list):
                logger.error(f"The 'features' attribute in {pk_file} is not a list.")
                continue
            
            list_file = os.path.splitext(pk_file)[0] + '.list'
            list_file_paths.append(list_file)

          
            with open(list_file, 'wb') as f:
                pickle.dump(features, f)
            total_samples += len(features)
            logger.info(f"Successfully converted {pk_file} to {list_file} with {len(features)} samples.")
        except Exception as e:
            logger.error(f"Failed to process {pk_file}: {e}")
    logger.info("All .pk files have been processed.")
    return list_file_paths, total_samples


def get_patch_info(patch_json_path):
    with open(patch_json_path, 'r') as f:
        patch_data = json.load(f)
    patch_files = patch_data.get('patch_files', [])
    patch_code = patch_data.get('patch_code', [])

    patch_dict = {}
    for patch in patch_code:
        file_path = patch.get('file')
        line = patch.get('line_number')
        if file_path and line:
            file_name = os.path.basename(file_path)  
            if file_name not in patch_dict:
                patch_dict[file_name] = set()
            patch_dict[file_name].add(line)
    return patch_dict


def process_vuldetect_example_dataset(dataset_dir, config):
    examples = []
    max_lines_per_segment = config.max_num_inst  # Maximum number of lines per code snippet
    min_lines_per_segment = config.min_num_inst  # Minimum number of lines per code snippet
    file_list = []
    for root, dirs, files in os.walk(dataset_dir):
        controlflow_files = [file for file in files if file.endswith("_ControlFlow.json")]
        file_list.extend(os.path.join(root, file) for file in controlflow_files)


    label_counts = {
        0: 0, 
        1: 0, 
        # 1: 0,  # stack_overflow
        # 2: 0,  # heap_overflow
        # 3: 0,  # overflowed_call
        # 4: 0,  # underflowed_call
        # 5: 0,  # second_free
        # 6: 0,  # use_after_free
    }

    Vul_label_File_counts = 0
    Fix_label_File_counts = 0
    Normal_label_File_counts = 0
    Other_label_File_counts = 0

    Vul_label_Example_counts = 0
    Normal_label_Example_counts = 0
    
    for controlflow_file in tqdm(file_list, desc="Processing files", unit="file"):

        base_name = controlflow_file.replace("_ControlFlow.json", "")
        try:
            with open(controlflow_file, "r") as f:
                data = json.load(f)
            
            func_labels = data.get("graph", {}).get("label", [])
            nodes = data.get("nodes", [])
            ircode = [
                re.sub(r', !dbg !\d+', '', node.get("irString", "").strip()) 
                for node in nodes
            ]
            node_labels = [node.get("label", "") for node in nodes]
            node_line_numbers = [node.get("line_number", "") for node in nodes]
            node_filenames = [node.get("filename", "") for node in nodes]
            node_ids = [node.get("id", -1) for node in nodes]  

            Linenum = len(ircode)
            func_slice_num = 0  
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: 文件 {controlflow_file} : {e}")
            continue
        except Exception as e:
            logger.error(f"err: file  {controlflow_file} : {e}")

            if Linenum <= min_lines_per_segment:
                continue
     

        func_label_ids = 0

        if 1 in func_labels:
            Vul_label_File_counts = Vul_label_File_counts + 1
            func_label_ids = 1
        elif 0 in func_labels:    
            Fix_label_File_counts = Fix_label_File_counts + 1
            func_label_ids = 0           
        elif any("Vul" in label for label in func_labels):
            Vul_label_File_counts = Vul_label_File_counts + 1
            func_label_ids = 1
        elif any("Normal" in label for label in func_labels):  
            Normal_label_File_counts = Normal_label_File_counts + 1
            func_label_ids = 0
        elif any("stack_overflow" in label for label in func_labels):
            func_label_ids = 1  # 存在 "stack_overflow"
        elif any("heap_overflow" in label for label in func_labels):
            func_label_ids = 1  # 存在 "heap_overflow"
        elif any("overflowed_call" in label for label in func_labels):
            func_label_ids = 1  # 存在 "overflowed_call"
        elif any("underflowed_call" in label for label in func_labels):
            func_label_ids = 1  # 存在 "underflowed_call"
        elif any("second_free" in label for label in func_labels):
            func_label_ids = 1  # 存在 "second_free"
        elif any("use_after_free" in label for label in func_labels):
            func_label_ids = 1  # 存在 "use_after_free"
        else:
            func_label_ids = 0 
            Other_label_File_counts = Other_label_File_counts + 1

        label_counts[func_label_ids] += 1
    
        logger.info(f"process_pretrain_example_dataset: {controlflow_file}, base_name:{base_name}, func_label_ids:{func_label_ids}")
        logger.info(f" Vul_label_File_counts:{Vul_label_File_counts}, Fix_label_File_counts:{Fix_label_File_counts}, Normal_label_File_counts:{Normal_label_File_counts}, Other_label_File_counts:{Other_label_File_counts}")


        # If the number of code lines is less than or equal to max_lines_per_segment
        if Linenum <= max_lines_per_segment:
            if func_label_ids == 1:
                Vul_label_Example_counts = Vul_label_Example_counts + 1
            elif func_label_ids == 0:
                Normal_label_Example_counts = Normal_label_Example_counts + 1
            src_nodeid = node_ids[0] if node_ids else -1
            dst_nodeid = node_ids[-1] if node_ids else -1
            # Create Example object.
            example = Example(
                base_filename=base_name,
                func_slice_num=func_slice_num,
                IRcode=ircode,
                src_nodeid=src_nodeid,
                dst_nodeid=dst_nodeid,
                Linenum=Linenum,
                label_idx=func_label_ids
                # label_idx=label_idx
            )
            examples.append(example)
        else: # When the number of code lines exceeds max_lines_per_segment
            window_size = config.inst_window_size
            segment_size = max_lines_per_segment  

            if func_label_ids > 0:
                start_id = 0
                while start_id + min_lines_per_segment <= Linenum:
                    end_id = min(start_id + max_lines_per_segment, Linenum)

                    ir_slice = ircode[start_id:end_id]
                    Linenum_slice = len(ir_slice)
                    src_nodeid = node_ids[start_id] 
                    dst_nodeid = node_ids[end_id - 1]
                    filename_base_name = os.path.basename(base_name)
                    if "omitgood" in filename_base_name:
                        label_present = any(
                            any("overflow" in label or "free" in label or "_call" in label for label in node_labels[idx])
                            for idx in range(start_id, end_id)
                        )
                        logger.info(f"omitgood filename_base_name :{filename_base_name}")
         
                    label_present = False
                    for idx in range(start_id, end_id):
                        if  "vul" in node_labels[idx]:
                            label_present = True
                            break 
                    logger.info(f" filename_base_name :{filename_base_name}, label_present:{label_present}")
                    if not label_present:
                        start_id += window_size
                        func_slice_num += 1
                        continue
                    logger.info(f"vul max_lines_per_segment base_name:{filename_base_name}, start_id:{start_id}, Linenum:{Linenum}, func_slice_num:{func_slice_num}")
                    example = Example(
                        base_filename=base_name,
                        func_slice_num=func_slice_num,
                        IRcode=ir_slice,
                        src_nodeid=src_nodeid,
                        dst_nodeid=dst_nodeid,
                        Linenum=Linenum_slice,
                        label_idx=func_label_ids
                    )
                    Vul_label_Example_counts = Vul_label_Example_counts + 1
                    examples.append(example)
                    start_id += window_size
                    func_slice_num += 1
            else: 
                start_id = 0
                while start_id + min_lines_per_segment <= Linenum:
                    end_id = min(start_id + segment_size, Linenum)
                    ir_slice = ircode[start_id:end_id]
                    Linenum_slice = len(ir_slice)
                    src_nodeid = node_ids[start_id] if start_id < Linenum else -1
                    dst_nodeid = node_ids[end_id - 1] if (end_id - 1) < Linenum else -1
                    example = Example(
                        base_filename=base_name,
                        func_slice_num=func_slice_num,
                        IRcode=ir_slice,
                        src_nodeid=src_nodeid,
                        dst_nodeid=dst_nodeid,
                        Linenum=Linenum_slice,
                        label_idx=0
                    )
                    logger.info(f"fix max_lines_per_segment base_name:{base_name}, start_id:{start_id}, Linenum:{Linenum}, func_slice_num:{func_slice_num}")
                    Normal_label_Example_counts = Normal_label_Example_counts + 1
                    examples.append(example)
                    start_id += window_size
                    func_slice_num += 1
    logger.info(f" Vul_label_Example_counts:{Vul_label_Example_counts}, Normal_label_Example_counts:{Normal_label_Example_counts}")
    log_file_path = 'example_dataset.log'
    with open(log_file_path, 'a') as log_file:
        for label_id, count in label_counts.items():
            log_file.write(f"Label {label_id}: {count} \n")
        log_file.write(f"Vul_label_Example_counts: {Vul_label_Example_counts}, Normal_label_Example_counts: {Normal_label_Example_counts} \n")
    return examples


def process_vulclassify_example_dataset(dataset_dir, config):
    examples = []
    max_lines_per_segment = config.max_num_inst 
    min_lines_per_segment = config.min_num_inst 
    file_list = []
    for root, dirs, files in os.walk(dataset_dir):
        controlflow_files = [file for file in files if file.endswith("_ControlFlow.json")]
        file_list.extend(os.path.join(root, file) for file in controlflow_files)


    label_counts = {
        0: 0,  # 0 non-vul
        1: 0,  # CWE121
        2: 0,  # CWE122
        3: 0,  # CWE190
        4: 0,  # CWE191
    }


    
    for controlflow_file in tqdm(file_list, desc="Processing files", unit="file"):
        base_name = controlflow_file.replace("_ControlFlow.json", "")
        Juliet_base_name = base_name.split("/")[-1] 
        
        try:
            with open(controlflow_file, "r") as f:
                data = json.load(f)
            nodes = data.get("nodes", [])
            ircode = [
                re.sub(r', !dbg !\d+', '', node.get("irString", "").strip()) 
                for node in nodes
            ]
            node_labels = [node.get("label", "") for node in nodes]
            node_line_numbers = [node.get("line_number", "") for node in nodes]
            node_filenames = [node.get("filename", "") for node in nodes]
            node_ids = [node.get("id", -1) for node in nodes]  

            Linenum = len(ircode)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: 文件 {controlflow_file} : {e}")
            continue
        except Exception as e:
            logger.error(f"err: file  {controlflow_file} : {e}")

            if Linenum <= min_lines_per_segment:
                continue
     
        func_label_ids = 0
        if "_omitbad" in Juliet_base_name:
            func_label_ids = 0
        elif Juliet_base_name.startswith("CWE121_") and "_omitgood" in Juliet_base_name:
            func_label_ids = 1
        elif Juliet_base_name.startswith("CWE122_") and "_omitgood" in Juliet_base_name:
            func_label_ids = 2
        elif Juliet_base_name.startswith("CWE190_") and "_omitgood" in Juliet_base_name:
            func_label_ids = 3
        elif Juliet_base_name.startswith("CWE191_") and "_omitgood" in Juliet_base_name:
            func_label_ids = 4
        
        label_counts[func_label_ids] += 1
        logger.info(f"process_pretrain_example_dataset: {controlflow_file}, base_name:{base_name}, func_label_ids:{func_label_ids}")
        logger.info(f"label_counts:{label_counts}")
        func_slice_num = 0 
        if Linenum <= max_lines_per_segment:
            src_nodeid = node_ids[0] if node_ids else -1
            dst_nodeid = node_ids[-1] if node_ids else -1
            example = Example(
                base_filename=base_name,
                func_slice_num=func_slice_num,
                IRcode=ircode,
                src_nodeid=src_nodeid,
                dst_nodeid=dst_nodeid,
                Linenum=Linenum,
                label_idx=func_label_ids
                # label_idx=label_idx
            )
            examples.append(example)
        else:
            window_size = config.inst_window_size
            segment_size = max_lines_per_segment 
            start_id = 0
            while start_id + min_lines_per_segment <= Linenum:
                end_id = min(start_id + segment_size, Linenum)
                ir_slice = ircode[start_id:end_id]
                Linenum_slice = len(ir_slice)
                src_nodeid = node_ids[start_id] if start_id < Linenum else -1
                dst_nodeid = node_ids[end_id - 1] if (end_id - 1) < Linenum else -1
                example = Example(
                    base_filename=base_name,
                    func_slice_num=func_slice_num,
                    IRcode=ir_slice,
                    src_nodeid=src_nodeid,
                    dst_nodeid=dst_nodeid,
                    Linenum=Linenum_slice,
                    label_idx=func_label_ids
                )
                logger.info(f"fix max_lines_per_segment base_name:{base_name}, start_id:{start_id}, Linenum:{Linenum}, func_slice_num:{func_slice_num}")
                examples.append(example)
                start_id += window_size
                func_slice_num += 1
    log_file_path = 'vulclassify_example_dataset.log'
    with open(log_file_path, 'a') as log_file:
        for label_id, count in label_counts.items():
            log_file.write(f"Label {label_id}: {count} \n")
        
    return examples


def process_pretrain_example_dataset(dataset_dir, config):
    examples = []
    max_lines_per_segment = config.max_num_inst  
    min_lines_per_segment = config.min_num_inst  
    file_list = []
    for root, dirs, files in os.walk(dataset_dir):
        controlflow_files = [file for file in files if file.endswith("_ControlFlow.json")]
        file_list.extend(os.path.join(root, file) for file in controlflow_files)

    label_counts = {
        0: 0,  
        1: 0,  
        # 1: 0,  # stack_overflow
        # 2: 0,  # heap_overflow
        # 3: 0,  # overflowed_call
        # 4: 0,  # underflowed_call
        # 5: 0,  # second_free
        # 6: 0,  # use_after_free

    }

    for controlflow_file in tqdm(file_list, desc="Processing files", unit="file"):
        base_name = controlflow_file.replace("_ControlFlow.json", "")
        try:
            with open(controlflow_file, "r") as f:
                data = json.load(f)
            

            func_labels = data.get("graph", {}).get("label", [])
            nodes = data.get("nodes", [])

            ircode = [
                re.sub(r', !dbg !\d+', '', node.get("irString", "").strip()) 
                for node in nodes
            ]
            node_labels = [node.get("label", "") for node in nodes]
            node_ids = [node.get("id", -1) for node in nodes] 
            Linenum = len(ircode)
            func_slice_num = 0  
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: 文件 {controlflow_file} : {e}")
            continue
        except Exception as e:
            logger.error(f"err: file  {controlflow_file} : {e}")

            if Linenum <= min_lines_per_segment:
                continue
     

        # func_label_ids = 1 if func_labels else 0

        func_label_ids = 0
        logger.info(f"process_pretrain_example_dataset: {controlflow_file}, base_name:{base_name}, func_label_ids:{func_label_ids}")
    
        if Linenum <= max_lines_per_segment:

            src_nodeid = node_ids[0] if node_ids else -1
            dst_nodeid = node_ids[-1] if node_ids else -1
            example = Example(
                base_filename=base_name,
                func_slice_num=func_slice_num,
                IRcode=ircode,
                src_nodeid=src_nodeid,
                dst_nodeid=dst_nodeid,
                Linenum=Linenum,
                label_idx=func_label_ids
                # label_idx=label_idx
            )
            examples.append(example)
        else: 
            window_size = config.inst_window_size
            segment_size = max_lines_per_segment  
            start_id = 0
            while start_id + min_lines_per_segment <= Linenum:
                end_id = min(start_id + segment_size, Linenum)
                ir_slice = ircode[start_id:end_id]
                Linenum_slice = len(ir_slice)
                src_nodeid = node_ids[start_id] if start_id < Linenum else -1
                dst_nodeid = node_ids[end_id - 1] if (end_id - 1) < Linenum else -1
                example = Example(
                    base_filename=base_name,
                    func_slice_num=func_slice_num,
                    IRcode=ir_slice,
                    src_nodeid=src_nodeid,
                    dst_nodeid=dst_nodeid,
                    Linenum=Linenum_slice,
                    label_idx=0
                )
                examples.append(example)
                start_id += window_size
                func_slice_num += 1

    log_file_path = 'example_dataset.log'
    with open(log_file_path, 'a') as log_file:
        for label_id, count in label_counts.items():
            log_file.write(f"Label {label_id}: {count} occurrences\n")
    return examples


# all_inst_encoder_input_ids: Token encodings for each instruction group.
# input_instcount_list: Number of instructions in each group.
# input_inst_tokencount: Number of tokens for each instruction.
# input_ids: Instruction mask inputs for the WelkIR model.
# cfg_matrix: Control Flow Graph matrix.
# dfg_matrix: Data Flow Graph matrix.
# rdg_matrix: Reachability Dependency Graph matrix.

def encode_example(example, config, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab):
    if config.task == "pretrain":
        model_inputs = prepare_model_inputs(
            config=config,
            example=example,
            tokenizer=tokenizer,
            cfg_vocab=cfg_vocab,
            dfg_vocab=dfg_vocab,
            rdg_vocab=rdg_vocab     
        )
        # logger.info("finish prepare_model_inputs")
        
        if model_inputs is None:
            return None
        input_feature = InputFeature(
            input_ids=torch.tensor(model_inputs["input_ids"], dtype=torch.int),
            all_inst_encoder_input_ids=torch.tensor(model_inputs["all_inst_encoder_input_ids"], dtype=torch.int),
            total_group_count=torch.tensor(model_inputs["total_group_count"], dtype=torch.int),
            input_instcount_list=torch.tensor(model_inputs["input_instcount_list"], dtype=torch.int),
            input_inst_tokencount=torch.tensor(model_inputs["input_inst_tokencount"], dtype=torch.int),
            cfg_matrix=torch.tensor(model_inputs["cfg_matrix"], dtype=torch.int),
            dfg_matrix=torch.tensor(model_inputs["dfg_matrix"], dtype=torch.int),
            rdg_matrix=torch.tensor(model_inputs["rdg_matrix"], dtype=torch.int),
            labels=torch.tensor([example.label_idx], dtype=torch.long),
            mask_all_inst_encoder_input_ids=None,
            inst_mlm_labels=None,
            cfg_matrix_mask=None,
            dfg_matrix_mask=None,
            rdg_matrix_mask=None
        )
        return input_feature
    elif config.task == "vuldetect":
       
        model_inputs = prepare_model_inputs(
            config=config,
            example=example,
            tokenizer=tokenizer,
            cfg_vocab=cfg_vocab,
            dfg_vocab=dfg_vocab,
            rdg_vocab=rdg_vocab     
        )
        logger.info(f"example.base_filename:{example.base_filename}")
        if model_inputs is None:
            return None
        VulDetect_feature = VulDetect_InputFeature(
            input_ids=torch.tensor(model_inputs["input_ids"], dtype=torch.int),
            all_inst_encoder_input_ids=torch.tensor(model_inputs["all_inst_encoder_input_ids"], dtype=torch.int),
            total_group_count=torch.tensor(model_inputs["total_group_count"], dtype=torch.int),
            input_instcount_list=torch.tensor(model_inputs["input_instcount_list"], dtype=torch.int),
            input_inst_tokencount=torch.tensor(model_inputs["input_inst_tokencount"], dtype=torch.int),
            cfg_matrix=torch.tensor(model_inputs["cfg_matrix"], dtype=torch.int),
            dfg_matrix=torch.tensor(model_inputs["dfg_matrix"], dtype=torch.int),
            rdg_matrix=torch.tensor(model_inputs["rdg_matrix"], dtype=torch.int),
            labels=torch.tensor([example.label_idx], dtype=torch.long),
            base_filename=example.base_filename,
            inst_mlm_labels=None,
            # add 20250323 attention
            func_slice_num=example.func_slice_num,
            src_nodeid=example.src_nodeid,
            dst_nodeid=example.dst_nodeid,
            ###########################
        )
        return VulDetect_feature
    elif config.task == "vulclassify":
        model_inputs = prepare_model_inputs(
            config=config,
            example=example,
            tokenizer=tokenizer,
            cfg_vocab=cfg_vocab,
            dfg_vocab=dfg_vocab,
            rdg_vocab=rdg_vocab     
        )
        logger.info(f"example.base_filename:{example.base_filename}")
        if model_inputs is None:
            return None
        VulDetect_feature = VulDetect_InputFeature(
            input_ids=torch.tensor(model_inputs["input_ids"], dtype=torch.int),
            all_inst_encoder_input_ids=torch.tensor(model_inputs["all_inst_encoder_input_ids"], dtype=torch.int),
            total_group_count=torch.tensor(model_inputs["total_group_count"], dtype=torch.int),
            input_instcount_list=torch.tensor(model_inputs["input_instcount_list"], dtype=torch.int),
            input_inst_tokencount=torch.tensor(model_inputs["input_inst_tokencount"], dtype=torch.int),
            cfg_matrix=torch.tensor(model_inputs["cfg_matrix"], dtype=torch.int),
            dfg_matrix=torch.tensor(model_inputs["dfg_matrix"], dtype=torch.int),
            rdg_matrix=torch.tensor(model_inputs["rdg_matrix"], dtype=torch.int),
            labels=torch.tensor([example.label_idx], dtype=torch.long),
            base_filename=example.base_filename,
            inst_mlm_labels=None,

        )
        return VulDetect_feature
  

def create_inst_matrix(dataset_dir, example, vocab, matrix_type, total_inst_count):
    suffix = {
        "cfg": "ControlFlow.json",
        "dfg": "DefUse.json",
        "rdg": "ReachDefinition.json"
    }.get(matrix_type.lower())
    
    if not suffix:
        raise ValueError(f"Invalid matrix_type: {matrix_type}")
        return None
    filepath = f"{example.base_filename}_{suffix}"
    # logger.info(f"create_inst_matrix filepath:{filepath}")

    total_inst_count
    #filepath = os.path.join(dataset_dir, filename)

    num_nodes = total_inst_count
    matrix = np.full((num_nodes, num_nodes), 1, dtype=int) 
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        

        link_key = {
            "cfg": "controlType",
            "dfg": "dtype",
            "rdg": "operation"
        }[matrix_type.lower()]
        #links = data.get("graph", {}).get("links", [])
        links = data.get("links", [])

        for link in links:
            source = link.get("source")
            target = link.get("target")
            relation_type = link.get(link_key, "other")
            # logger.info(f"link_key:{link_key} , relation_type: {relation_type}")

            if example.src_nodeid <= source <=  (example.src_nodeid + total_inst_count - 1)  and example.src_nodeid <= target <= (example.src_nodeid + total_inst_count - 1) :
                src_idx = source - example.src_nodeid
                tgt_idx = target - example.src_nodeid        
                matrix_value = vocab.get(relation_type, vocab.get("other", -1))    
                # logger.info(f"link_key:{link_key} , matrix_value: {matrix_value}")
                matrix[src_idx][tgt_idx] = matrix_value
    except FileNotFoundError:
        logger.error(f"Warning: File {filepath} not found. Returning a matrix filled with 1.")
        return None
    
    return matrix


def pad_or_truncate_matrix(matrix, target_size, padding_id=0, sampling_rate=0.15):
    padded = np.full(shape=target_size, fill_value=padding_id, dtype=np.int32)
    matrix_mask = np.zeros(shape=target_size, dtype=np.int32)
    num_rows = min(matrix.shape[0], target_size[0] - 2)
    num_cols = min(matrix.shape[1], target_size[1] - 2)
    padded[1:num_rows + 1, 1:num_cols + 1] = matrix[:num_rows, :num_cols]
    matrix_mask[1:num_rows + 1, 1:num_cols + 1] = 1
    # Randomly sample 15% of the valid positions.
    valid_indices = np.argwhere(matrix_mask == 1)
    num_samples =  int(len(valid_indices) * sampling_rate) + 1
    sampled_indices = np.random.choice(len(valid_indices), size=num_samples, replace=False)
    sampled_matrix_mask = np.zeros(shape=target_size, dtype=np.int32)
    for idx in sampled_indices:
        sampled_matrix_mask[tuple(valid_indices[idx])] = 1
    return padded, sampled_matrix_mask


def prepare_model_inputs(config, example, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab):
    max_token_per_group = config.max_token_per_group
    example_input_strings = []
    input_instcount_list = []  
    input_inst_tokencount = [] 
    current_group = []
    current_token_count = 0
    current_inst_count = 0
    current_inst_tokencount = []

    total_group_count = 0
    max_total_group_count = config.max_group
    total_inst_count = 0
    for idx, inst in enumerate(example.IRcode):
        tokens = tokenizer.tokenize(inst)
        token_count = len(tokens)
        token_count = token_count + 1 

        if current_token_count + token_count  <= max_token_per_group and current_inst_count < config.group_max_inst_count -2:
            current_group.append(inst)
            current_token_count += token_count
            current_inst_count += 1
            current_inst_tokencount.append(token_count)
        else:

            logger.info(f"group current_inst_count :{current_inst_count}")
            example_input_strings.append(f"{'</s>'.join(current_group)}")
            input_instcount_list.append(current_inst_count)
            input_inst_tokencount.append(current_inst_tokencount)
            total_group_count = total_group_count + 1
            total_inst_count = total_inst_count + current_inst_count

            if(total_group_count > max_total_group_count-1): 
                break
            current_group = [inst]
            current_token_count = token_count
            current_inst_count = 1
            current_inst_tokencount = [token_count]


    if current_group and total_group_count < max_total_group_count:
        logger.info(f"group current_inst_count :{current_inst_count}") 
        example_input_strings.append(f"{'</s>'.join(current_group)}")
        input_instcount_list.append(current_inst_count)
        input_inst_tokencount.append(current_inst_tokencount)
        total_group_count = total_group_count + 1
        total_inst_count = total_inst_count + current_inst_count

    logger.info(f"inst example.IRcode count:{len(example.IRcode)}, total_inst_count:{total_inst_count}")
    logger.info(f"total_group_count:{total_group_count}") 
             
    padded_input_inst_tokencount = [
        group + [-1] * (config.group_max_inst_count - len(group)) for group in input_inst_tokencount
    ]

    all_inst_encoder_input_ids = []
    for input_string in example_input_strings:
        encoded_inputs = tokenizer.encode(
            input_string,
            padding="max_length",
            truncation=True,
            max_length=config.max_inst_input_length,
            add_special_tokens=True
        )

        all_inst_encoder_input_ids.append(encoded_inputs)  # 保存编码后的token id
    

    if total_group_count < max_total_group_count:
        pad_group = [config.pad_token_id] * config.max_inst_input_length
        all_inst_encoder_input_ids.extend([pad_group] * (max_total_group_count - total_group_count))
        input_instcount_list.extend([0] * (max_total_group_count - total_group_count))
        pad_inst_tokencount_group = [-1] * config.group_max_inst_count
        padded_input_inst_tokencount.extend([pad_inst_tokencount_group] * (max_total_group_count - total_group_count))
    elif total_group_count > max_total_group_count:
        logger.error(f"total_group_count > max_total_group_count" )
        sys.exit()


    inst_padding_length = config.max_num_inst - total_inst_count
    inst_input_ids = (
            [tokenizer.bos_token_id]  +
            [tokenizer.mask_token_id] * total_inst_count   # example.Linenum
            + [tokenizer.eos_token_id]
            + [tokenizer.pad_token_id] * inst_padding_length
            
    )

    dataset_dir = os.path.join(config.dataset_root, "pretrain")
    cfg_matrix = create_inst_matrix(dataset_dir ,example, cfg_vocab, "cfg", total_inst_count)
    dfg_matrix = create_inst_matrix(dataset_dir ,example, dfg_vocab, "dfg", total_inst_count)
    rdg_matrix = create_inst_matrix(dataset_dir ,example, rdg_vocab, "rdg", total_inst_count)

    if rdg_matrix is None or dfg_matrix is None or rdg_matrix is None:
        return None
    
    padded_cfg, cfg_matrix_mask = pad_or_truncate_matrix(
        cfg_matrix, target_size=(config.max_num_inst+2, config.max_num_inst+2)
    )
    padded_dfg, dfg_matrix_mask = pad_or_truncate_matrix(
        dfg_matrix, target_size=(config.max_num_inst+2, config.max_num_inst+2)
    )
    padded_rdg, rdg_matrix_mask = pad_or_truncate_matrix(
        rdg_matrix, target_size=(config.max_num_inst+2, config.max_num_inst+2)
    )
    
    model_inputs = {
        "all_inst_encoder_input_ids": all_inst_encoder_input_ids,
        "total_group_count": total_group_count,
        "input_instcount_list": input_instcount_list,
        "input_inst_tokencount": padded_input_inst_tokencount,
        "input_ids": inst_input_ids,
        "cfg_matrix": padded_cfg,
        "dfg_matrix": padded_dfg,
        "rdg_matrix": padded_rdg,
        "cfg_matrix_mask": cfg_matrix_mask,
        "dfg_matrix_mask": dfg_matrix_mask,
        "rdg_matrix_mask": rdg_matrix_mask,
    }
    return model_inputs


def multiprocess_func(func, items, num_processors=None, single_thread=False) -> List:

    if items is None:
        logger.error(f"The `items` parameter cannot be None. Please provide a valid list of items.")
        return []
    elif len(items) == 0:
        logger.error("Warning: The `items` list is empty. Returning an empty result.")
        return []

    processes = num_processors if num_processors else multiprocessing.cpu_count()

    if processes > 1 and not single_thread:
        with multiprocessing.Pool(processes=processes) as p:
            results = list(p.map(func, tqdm(items, total=len(items), ascii=True)))
    else:
        results = [func(item) for item in tqdm(items, total=len(items), ascii=True)]
    return results


def convert_data_to_features(
    config,
    examples: List[Example],
    tokenizer: PreTrainedTokenizerFast,
    cfg_vocab: dict,
    dfg_vocab: dict,
    rdg_vocab: dict 
) -> List[Optional[InputFeature]]:
    encode_func = partial(
        encode_example,
        config=config,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab
    )

    features = multiprocess_func(encode_func, examples, single_thread=config.single_thread_parsing)
    features = [feature for feature in features if feature is not None]
    return features



def pkfile_process_and_save_data_chunk(
    config,
    examples_chunk,
    tokenizer,
    cfg_vocab,
    dfg_vocab,
    rdg_vocab,
    output_path,
):

    features = convert_data_to_features(
        config=config,
        examples=examples_chunk,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab
    )

    logger.info(f"Processed {len(examples_chunk)} examples, saving to {output_path}.")
    # dataset = PretrainDataset(features)
    # save_pickle(obj=dataset, path=output_path)

    save_pickle(obj=features, path=output_path)
    logger.info(f"Finished saving dataset to {output_path}.")



def pkfile_split_examples(examples, num_processes):
    """将 examples 列表分成 num_processes 份"""
    chunk_size = len(examples) // num_processes
    chunks = [examples[i:i + chunk_size] for i in range(0, len(examples), chunk_size)]
    return chunks

def pkfile_create_pretrain_datasets_in_parallel(
    config,
    split,
    examples,
    tokenizer,
    cfg_vocab,
    dfg_vocab,
    rdg_vocab,
    dataset_dir,
    num_processes
):
    chunks = pkfile_split_examples(examples, num_processes)

    processes = []
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(dataset_dir, "cache", f"{config.task}-{split}-dataset-{i+1:02d}.pk")
        p = multiprocessing.Process(
            target=pkfile_process_and_save_data_chunk,
            args=(config, chunk, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, output_path)
        )
        processes.append(p)
    
    for p in processes:
        p.start()
        p.join()
    
    logger.info("All processes finished, datasets have been saved.")



def prepare_pretrain_dataset_to_pkfile (
    config, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, split, shuffle=True
):

    dataset_dir = config.dataset_root
    IR_Ptrtrain_Graph_folder = config.IR_Ptrtrain_Graph_folder     
    logger.info(f"Start preparing {split} data...")

    examples_cache_path = os.path.join(
        IR_Ptrtrain_Graph_folder, "cache", f"{config.task}-{split}-examples.pk"
    )
    
    pretrain_dataset_path = os.path.join(dataset_dir, "pretrain")

    logger.info(f" start prepare_pretrain_dataset_to_pkfile ... ")
    logger.info(f"IR_Ptrtrain_Graph_folder: {IR_Ptrtrain_Graph_folder}...")
    logger.info(f"dataset_dir: {dataset_dir}...")
    logger.info(f" examples_cache_path: {examples_cache_path} ")
    logger.info(f" pretrain_dataset_path : {pretrain_dataset_path}")

    logger.info(f" start prepare_pretrain_dataset  load examples ... ")
    # load examples
    if os.path.exists(examples_cache_path):
        # load parsed examples from cache
        logger.info(f"Find cache of examples: {examples_cache_path}")
        examples = load_pickle(examples_cache_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logger.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        examples = process_pretrain_example_dataset(IR_Ptrtrain_Graph_folder, config)
        if shuffle:
            random.shuffle(examples) 
        logger.info("save examples pickle...")
        save_pickle(obj=examples, path=examples_cache_path)

    logger.info(f" start prepare_pretrain_dataset  load examples: {len(examples)},  ... ")
    pkfile_create_pretrain_datasets_in_parallel(
        config=config,
        split=split,
        examples=examples,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab,
        dataset_dir=pretrain_dataset_path,
        num_processes=40 
    )
    return 



def prepare_pretrain_dataset(
    config, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, split
) -> Tuple[PretrainDataset, List[Example]]:
    # check split
    #assert split in ["pretrain", "train", "valid", "test"]

    dataset_dir = os.path.join(config.dataset_root, split)
    logger.info(f"Start preparing {split} data...")

    # dataset_cache_path = os.path.join(
    #     dataset_dir, "cache", f"{config.task}-{split}-dataset.pk"
    # )
    dataset_cache_path = os.path.join(
        dataset_dir, "cache", "pretrain-CWE_all-dataset.pk")

    examples_cache_path = os.path.join(
        dataset_dir, "cache", f"{config.task}-{split}-examples.pk"
    )
    logger.info(f" start prepare_pretrain_dataset  load dataset ... ")
    # load dataset
    # if config.use_data_cache and os.path.exists(dataset_cache_path):
    #     # load parsed dataset from cache
    #     logger.info(f"Find cache of dataset: {dataset_cache_path}")
    #     dataset = load_pickle(dataset_cache_path)
    #     assert isinstance(dataset, PretrainDataset)
    #     logger.info(f"Dataset is loaded from cache")
    #     return dataset



    logger.info(f" start prepare_pretrain_dataset  load examples ... ")
    # load examples
    if config.use_data_cache and os.path.exists(examples_cache_path):
        # load parsed examples from cache
        logger.info(f"Find cache of examples: {examples_cache_path}")
        examples = load_pickle(examples_cache_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logger.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logger.info("Start loading split metadata...")
        examples = process_pretrain_example_dataset(dataset_dir, config)
        # save as cache if needed
        if config.use_data_cache:
            save_pickle(obj=examples, path=examples_cache_path)


    logger.info(f" start prepare_pretrain_dataset  load dataset ... ")
    # load dataset
    if config.use_data_cache and os.path.exists(dataset_cache_path):
        # load parsed dataset from cache
        logger.info(f"Find cache of dataset: {dataset_cache_path}")
        dataset = load_pickle(dataset_cache_path)
        assert isinstance(dataset, PretrainDataset)
        logger.info(f"Dataset is loaded from cache")
    else:
        logger.info("Start parsing and tokenizing...")
        features = convert_data_to_features(
            config=config,
            examples=examples,
            tokenizer=tokenizer,
            cfg_vocab=cfg_vocab,
            dfg_vocab=dfg_vocab,
            rdg_vocab=rdg_vocab
        )
        logger.info("finish convert_data_to_features.")

        dataset = PretrainDataset(features)

        if config.use_data_cache:
            save_pickle(obj=dataset, path=dataset_cache_path)
        logger.info("finish save_pickle dataset_cache_path.")
    return dataset


def remove_special_tokens(s, tokenizer) -> str:
    """Removes all special tokens from given str."""
    for token in tokenizer.all_special_tokens:
        s = s.replace(token, " ")
    return s


def tokenize(txt, tokenizer, max_length) -> List[int]:
    """Converts code to input ids, only for code."""
    source = remove_special_tokens(txt, tokenizer)
    return tokenizer.encode(
        source, padding="max_length", max_length=max_length, truncation=True
    )


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass

class MixedPrecisionManager(object):
    def __init__(self, activated):
        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if isinstance(loss, torch.Tensor) and loss.ndimension() > 0:
            loss = loss.mean()  # 对多卡情况进行平均，确保 loss 是一个标量
            # loss = torch.mean(loss) 
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, model, optimizer, max_grad_norm=None):
        if self.activated:
            if max_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()



class EarlyStopController(object):
    """
    A controller for early stopping.
    Args:
        patience (int):
            Maximum number of consecutive epochs without breaking the best record.
        higher_is_better (bool, optional, defaults to True):
            Whether a higher record is seen as a better one.
    """

    def __init__(self, patience: int, best_model_dir, higher_is_better=True):
        self.patience = patience
        self.best_model_dir = best_model_dir
        self.higher_is_better = higher_is_better

        self.early_stop = False
        self.hit = False
        self.counter = 0

        self.best_score = None
        self.best_state = {"epoch": None, "step": None}

    def __call__(self, score: float, model, epoch: int, step: int):
        """Calls this after getting the validation metric each epoch."""
        # first calls
        if self.best_score is None:
            self.__update_best(score, model, epoch, step)
        else:
            # not hits the best record
            if (self.higher_is_better and score < self.best_score) or (
                not self.higher_is_better and score > self.best_score
            ):
                self.hit = False
                self.counter += 1
                if self.counter > self.patience:
                    self.early_stop = True
            # hits the best record
            else:
                self.__update_best(score, model, epoch, step)

    def __update_best(self, score, model, epoch, step):
        self.best_score = score
        self.hit = True
        self.counter = 0
        self.best_state["epoch"] = epoch
        self.best_state["step"] = step

        os.makedirs(self.best_model_dir, exist_ok=True)
        torch.save(
            model.module if isinstance(model, torch.nn.DataParallel) else model,
            os.path.join(self.best_model_dir, "model.pt")
        )

        if hasattr(model, "welkir_model"):
            model.welkir_model.save_pretrained(
                os.path.join(self.best_model_dir, "WelkirModel")
            )

    def load_best_model(self):
        load_path = os.path.join(self.best_model_dir, "model.pt")
        obj = None
        if os.path.exists(load_path):
            obj = torch.load(load_path, map_location=torch.device("cpu"))
        return obj

    def get_best_score(self):
        return self.best_score


# https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer_pt_utils.py#L473
@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False):
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (
            num_active_elements * log_probs.shape[-1]
        )
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss



class VuldetectDataset(Dataset):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        # return_dict = self.features[item].__dict__
        # print({k: v.size() for k, v in return_dict.items()})
        feature = self.features[item]
        labels = feature.labels if feature.labels is not None else 0
        return {
            'input_ids': feature.input_ids,
            'all_inst_encoder_input_ids': feature.all_inst_encoder_input_ids,
            'input_instcount_list': feature.input_instcount_list,
            'input_inst_tokencount': feature.input_inst_tokencount,
            'cfg_matrix': feature.cfg_matrix,
            'dfg_matrix': feature.dfg_matrix,
            'rdg_matrix': feature.rdg_matrix,
            'total_group_count': feature.total_group_count,
            'labels': labels,
        }

    def __len__(self):
        return len(self.features)




def split_dataset_with_train(dataset, train_ratio=0.8, validation_ratio=0.1, random_state=42):
    dataset = shuffle(dataset, random_state=random_state)
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    validation_size = int(total_size * validation_ratio)
    test_size = total_size - train_size - validation_size

    train_set = dataset[:train_size]
    validation_set = dataset[train_size:train_size + validation_size]
    test_set = dataset[train_size + validation_size:]

    logger.info(f"Total number of datasets:{total_size}")
    logger.info(f"training set:{train_size},  validation_set:{validation_size}, testing set:{test_size}")
    return train_set, validation_set, test_set


    
def pkfile_create_vuldetect_datasets_in_parallel(
    config,
    split,
    examples,
    tokenizer,
    cfg_vocab,
    dfg_vocab,
    rdg_vocab,
    dataset_dir,
    num_processes
):

    chunks = pkfile_split_examples(examples, num_processes)
    processes = []
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(dataset_dir, "cache", f"{config.task}-{split}-dataset-{i+1:02d}.pk")
        p = multiprocessing.Process(
            target=pkfile_process_and_save_data_chunk,
            args=(config, chunk, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, output_path)
        )
        processes.append(p)
    for p in processes:
        p.start()
        p.join()

    logger.info("All processes finished, datasets have been saved.")

# dataset_train_path = os.path.join(config.dataset_root, "CWE121/train", "train-CWE121-dataset.pk")
def prepare_vuldetect_dataset(
    config, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, split, shuffle=True
) -> Tuple[PretrainDataset, List[Example]]:
    # check split
    #assert split in ["pretrain", "train", "valid", "test"]

    logger.info(f"Start preparing {split} data...")
    dataset_dir = config.dataset_root


    IR_Train_Graph_folder = config.IR_Train_Graph_folder
    IR_Validation_Graph_folder = config.IR_Validation_Graph_folder
    IR_Test_Graph_folder = config.IR_Test_Graph_folder

    train_examples_path = os.path.join(IR_Train_Graph_folder, "cache", f"train-examples.pk")
    train_dataset_path = os.path.join(dataset_dir, "train")

    validation_examples_path = os.path.join(IR_Validation_Graph_folder, "cache", f"validation-examples.pk")
    validation_dataset_path = os.path.join(dataset_dir, "validation")

    test_examples_path = os.path.join(IR_Test_Graph_folder, "cache", f"test-examples.pk")
    test_dataset_path = os.path.join(dataset_dir, "test")



    # logger.info(f" start prepare_vuldetect_dataset  load  train examples ... ")
    # # load train examples
    # if config.use_data_cache and os.path.exists(train_examples_path):
    #     # load parsed examples from cache
    #     logger.info(f"Find cache of examples: {train_examples_path}")
    #     examples = load_pickle(train_examples_path)
    #     assert isinstance(examples, list)
    #     assert isinstance(examples[0], Example)
    #     logger.info(f"Examples are loaded from cache")
    # else:
    #     # load and parse examples from disk files
    #     logger.info("Start train examples ...")
    #     examples = process_vuldetect_example_dataset(IR_Train_Graph_folder, config)
    #     # if shuffle:
    #     random.shuffle(examples) 
    #     # save as cache if needed
    #     if config.use_data_cache:
    #         save_pickle(obj=examples, path=train_examples_path)
    # logger.info(f" finish prepare_pretrain_dataset train load examples: {len(examples)},  ... ")
    # logger.info(f" start pkfile_create_vuldetect_datasets_in_parallel  train datasets ... ")
    # pkfile_create_vuldetect_datasets_in_parallel(
    #     config=config,
    #     split="train",
    #     examples=examples,
    #     tokenizer=tokenizer,
    #     cfg_vocab=cfg_vocab,
    #     dfg_vocab=dfg_vocab,
    #     rdg_vocab=rdg_vocab,
    #     dataset_dir=train_dataset_path,
    #     num_processes=8
    # )
    # logger.info(f" finish pkfile_create_vuldetect_datasets_in_parallel  train datasets ... ")



    logger.info(f" start prepare_vuldetect_dataset  load  validation examples ... ")
    # load validation examples
    if config.use_data_cache and os.path.exists(validation_examples_path):
        # load parsed examples from cache
        logger.info(f"Find cache of examples: {validation_examples_path}")
        examples = load_pickle(validation_examples_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logger.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logger.info("Start validation examples ...")
        examples = process_vuldetect_example_dataset(IR_Validation_Graph_folder, config)
        # save as cache if needed
        if config.use_data_cache:
            save_pickle(obj=examples, path=validation_examples_path)
    logger.info(f" finish prepare_pretrain_dataset  validation examples: {len(examples)},  ... ")
    logger.info(f" start pkfile_create_vuldetect_datasets_in_parallel  validation datasets ... ")
    pkfile_create_vuldetect_datasets_in_parallel(
        config=config,
        split="validation",
        examples=examples,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab,
        dataset_dir=validation_dataset_path,
        num_processes=4 
    )
    logger.info(f" finish pkfile_create_vuldetect_datasets_in_parallel  validation datasets ... ")


    logger.info(f" start prepare_vuldetect_dataset  load  test examples ... ")
    # load test examples
    if config.use_data_cache and os.path.exists(test_examples_path):
        # load parsed examples from cache
        logger.info(f"Find cache of examples: {test_examples_path}")
        examples = load_pickle(test_examples_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logger.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logger.info("Start test examples ...")
        examples = process_vuldetect_example_dataset(IR_Test_Graph_folder, config)
        # save as cache if needed
        if config.use_data_cache:
            save_pickle(obj=examples, path=test_examples_path)
    logger.info(f" finish prepare_pretrain_dataset  load test examples: {len(examples)},  ... ")
    logger.info(f" start pkfile_create_vuldetect_datasets_in_parallel  test datasets ... ")
    pkfile_create_vuldetect_datasets_in_parallel(
        config=config,
        split="test",
        examples=examples,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab,
        dataset_dir=test_dataset_path,
        num_processes=4 
    )
    logger.info(f" finish pkfile_create_vuldetect_datasets_in_parallel  test datasets ... ")



# dataset_train_path = os.path.join(config.dataset_root, "CWE121/train", "train-CWE121-dataset.pk")
def prepare_vulclassify_dataset(
    config, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, split, shuffle=True
) -> Tuple[PretrainDataset, List[Example]]:
    # check split
    #assert split in ["pretrain", "train", "valid", "test"]

    logger.info(f"Start preparing {split} data...")
    dataset_dir = config.dataset_root


    IR_Train_Graph_folder = config.IR_Train_Graph_folder
    IR_Validation_Graph_folder = config.IR_Validation_Graph_folder
    IR_Test_Graph_folder = config.IR_Test_Graph_folder

    train_examples_path = os.path.join(IR_Train_Graph_folder, "cache", f"train-examples.pk")
    train_dataset_path = os.path.join(dataset_dir, "train")

    validation_examples_path = os.path.join(IR_Validation_Graph_folder, "cache", f"validation-examples.pk")
    validation_dataset_path = os.path.join(dataset_dir, "validation")

    test_examples_path = os.path.join(IR_Test_Graph_folder, "cache", f"test-examples.pk")
    test_dataset_path = os.path.join(dataset_dir, "test")


    logger.info(f" start prepare_vulclassify_dataset  load  train examples ... ")
    # load train examples
    if config.use_data_cache and os.path.exists(train_examples_path):
        # load parsed examples from cache
        logger.info(f"Find cache of examples: {train_examples_path}")
        examples = load_pickle(train_examples_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logger.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logger.info("Start train examples ...")
        examples = process_vulclassify_example_dataset(IR_Train_Graph_folder, config)
        # if shuffle:
        random.shuffle(examples) 
        # save as cache if needed
        if config.use_data_cache:
            save_pickle(obj=examples, path=train_examples_path)
    logger.info(f" finish prepare_pretrain_dataset train load examples: {len(examples)},  ... ")
    logger.info(f" start prepare_vulclassify_dataset  train datasets ... ")
    pkfile_create_vuldetect_datasets_in_parallel(
        config=config,
        split="train",
        examples=examples,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab,
        dataset_dir=train_dataset_path,
        num_processes=8 
    )
    logger.info(f" finish pkfile_create_vuldetect_datasets_in_parallel  train datasets ... ")



    logger.info(f" start prepare_vulclassify_dataset  load  validation examples ... ")
    # load validation examples
    if config.use_data_cache and os.path.exists(validation_examples_path):
        # load parsed examples from cache
        logger.info(f"Find cache of examples: {validation_examples_path}")
        examples = load_pickle(validation_examples_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logger.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logger.info("Start validation examples ...")
        examples = process_vulclassify_example_dataset(IR_Validation_Graph_folder, config)
        # save as cache if needed
        if config.use_data_cache:
            save_pickle(obj=examples, path=validation_examples_path)
    logger.info(f" finish prepare_pretrain_dataset  validation examples: {len(examples)},  ... ")
    logger.info(f" start pkfile_create_vuldetect_datasets_in_parallel  validation datasets ... ")
    pkfile_create_vuldetect_datasets_in_parallel(
        config=config,
        split="validation",
        examples=examples,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab,
        dataset_dir=validation_dataset_path,
        num_processes=4 
    )
    logger.info(f" finish pkfile_create_vuldetect_datasets_in_parallel  validation datasets ... ")


    logger.info(f" start prepare_vulclassify_dataset  load  test examples ... ")
    # load test examples
    if config.use_data_cache and os.path.exists(test_examples_path):
        # load parsed examples from cache
        logger.info(f"Find cache of examples: {test_examples_path}")
        examples = load_pickle(test_examples_path)
        assert isinstance(examples, list)
        assert isinstance(examples[0], Example)
        logger.info(f"Examples are loaded from cache")
    else:
        # load and parse examples from disk files
        logger.info("Start test examples ...")
        examples = process_vulclassify_example_dataset(IR_Test_Graph_folder, config)
        # save as cache if needed
        if config.use_data_cache:
            save_pickle(obj=examples, path=test_examples_path)
    logger.info(f" finish prepare_pretrain_dataset  load test examples: {len(examples)},  ... ")
    logger.info(f" start pkfile_create_vuldetect_datasets_in_parallel  test datasets ... ")
    pkfile_create_vuldetect_datasets_in_parallel(
        config=config,
        split="test",
        examples=examples,
        tokenizer=tokenizer,
        cfg_vocab=cfg_vocab,
        dfg_vocab=dfg_vocab,
        rdg_vocab=rdg_vocab,
        dataset_dir=test_dataset_path,
        num_processes=4 
    )
    logger.info(f" finish pkfile_create_vuldetect_datasets_in_parallel  test datasets ... ")









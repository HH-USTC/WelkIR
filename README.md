# WelkIR

## Overview
In this repository you will find a Python implementation of WelkIR, a two-tier IR-based Transformer model for vulnerability detection, which is fine-tuned from a flow-sensitive pre-trained model on task-specific data.

## ARVul Collection
We construct ARVul, a vulnerability dataset based on LLVM IR. The dataset is compiled from 218 real-world software projects and comprises 3,732 vulnerable functions, which can be processed through the LLVM toolchain for optimization and static analysis.
PYTHONPATH="." python ARVul_Collection/run_arvo_compile.py 
The ARVul dataset can be accessed via the following link: (https://drive.google.com/drive/folders/1ZJ-MY-u7-XthTGw_Lnt5VhJGk44ByGx9?usp=sharing)

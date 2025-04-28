# WelkIR

## Overview
In this repository you will find a Python implementation of WelkIR, a two-tier IR-based Transformer model for vulnerability detection, which is fine-tuned from a flow-sensitive pre-trained model on task-specific data.

## ARVul Collection
We construct ARVul, a vulnerability dataset based on LLVM IR. The dataset is compiled from 218 real-world software projects and comprises 3,732 vulnerable functions, which can be processed through the LLVM toolchain for optimization and static analysis.
PYTHONPATH="." python ARVul_Collection/run_arvo_compile.py 
The ARVul dataset can be accessed via the following link: (https://drive.google.com/drive/folders/1ZJ-MY-u7-XthTGw_Lnt5VhJGk44ByGx9?usp=sharing)

##  Generate the Property Graphs
WelkIR_Dataprocess is an LLVM pass that generates Control Flow Graphs (CFGs), Def-Use Flow Graphs (DFGs), and Reaching Definition Graphs (RDGs) for the IR program.
e.g.  opt -load llvm-project-release-12.x/llvm-build/lib/LLVM_WelkIR.so -WelkIR -labelFilename={json_label_file} -OutputPath={output_graph_path} -IRFile_Type={IRFile_Type} ll_file_path -enable-new-pm=0


##  Evaluation
PYTHONPATH="." python main.py

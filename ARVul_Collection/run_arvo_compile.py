import json
import argparse
import shlex
import os
import subprocess
import queue
import random
import sys
import threading
import time
import yaml
import json
import argparse
import re
import chardet
from loguru import logger
from multiprocessing import Pool
from functools import partial
from unidiff import PatchSet
from args import add_args


logger.add("WelkIR_Dataprocess_info.log", rotation="10 MB", encoding="utf-8", enqueue=True)

SOURCE_FILE_EXTENSIONS = {'.c', '.cpp', '.cc', '.cxx'}
G_Trouble_Projects = [
]

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


# step1: parse_diff_file:Analyze patch files and extract function names
def parse_diff_file(diff_file_path):
    try:
        with open(diff_file_path, 'rb') as diff_file:
            raw_data = diff_file.read()
        if not raw_data:
            logger.warning(f"The file is empty: {diff_file_path}")
            return None
        detected_encoding = chardet.detect(raw_data)['encoding']
        if not detected_encoding:
            logger.error(f"Unable to detect file encoding: {diff_file_path}")
            return None
        diff_content = raw_data.decode(detected_encoding)
        patch = PatchSet(diff_content)
    except UnicodeDecodeError as e:
        error_position = e.start
        with open(diff_file_path, 'rb') as diff_file:
            raw_data = diff_file.read()
            error_data = raw_data[:error_position]
            error_line_number = error_data.count(b'\n') + 1
        logger.error(f"File decoding error occurred in file {diff_file_path}:{error_line_number}")
        return None
    except Exception as e:
        error_message = f"parsing the patch file {e}: {diff_file_path}"
        logger.error(error_message)
        return None
    
    patch_files = set()
    patch_code = []
    
    for patched_file in patch:
        try:
            file_path = patched_file.path
            if not file_path:
                logger.error(f"The file path is None or empty: {diff_file_path}")
                continue
            
            # Only process source code files
            _, ext = os.path.splitext(file_path.lower())
            if not ext in SOURCE_FILE_EXTENSIONS:
                # logger.info(f"Non source code files: {file_path}")
                continue            
            patch_files.add(file_path)

            for hunk in patched_file:
                for idx, line in enumerate(hunk):
                    if line.is_added or line.is_removed:
                        line_number = line.target_line_no if line.is_added else line.source_line_no
                        code_info = {
                            'file': file_path,
                            'line_number': line_number,
                            'code': line.value.rstrip('\n'),
                            'operation': 'addition' if line.is_added else 'deletion',
                        }
                        patch_code.append(code_info)
        except Exception as e:
            logger.error(f"hunt error occurred : {e}, diff_file_path: {diff_file_path} ")
            continue

    if not patch_files and not patch_code:
        logger.warning(f" no valid content in the patch file: {diff_file_path}")
        return None
    
    return {
        'patch_files': list(patch_files),
        'patch_code': patch_code
    }

# step1: Analyze patch files and extract function names
def process_Patches_diff_file(diff_file, config):
    os.makedirs(config["CollectLabels_Folder"], exist_ok=True)
    meta_name = diff_file.split('.')[0]
    Patches_output_file = os.path.join(os.path.abspath(config["CollectLabels_Folder"]), f"{meta_name}.json")
    diff_file_fullpath = os.path.join(os.path.abspath(config["Patches_Folder"]), diff_file)
    # logger.info(f"start process_Patches_json_file:{diff_file_fullpath}, Patches_output_file:{Patches_output_file}")

    parse_diff_result = parse_diff_file(diff_file_fullpath)
    if parse_diff_result:
        with open(Patches_output_file, 'w') as f:
            json.dump(parse_diff_result, f, indent=4)


def check_docker_user(container_id):
    command = f"docker exec {container_id} whoami"
    try:
        user = subprocess.check_output(command, shell=True, text=True).strip()
        logger.info(f"current user: {user}")
        return user
    except subprocess.CalledProcessError as e:
        logger.error(f"user fail: {e.output}")
        return None
    

def check_file_in_container(container_id, path):
    try:
        # Construct the command to check if the path is a file or directory
        command = (
            f"if [ -f {path} ]; then "
            f"echo 'file'; "
            f"elif [ -d {path} ]; then "
            f"echo 'directory'; "
            f"else "
            f"echo 'not found'; "
            f"fi"
        )
        
        # Execute the command in the specified container
        result = subprocess.run(
            ['docker', 'exec', '-u', 'root', container_id, 'sh', '-c', command],
            capture_output=True,
            text=True
        )
        
        # Check the output and return the result
        output = result.stdout.strip()
        if output == 'file':
            return True
        elif output == 'directory':
            return True
        elif output == 'not found':
            return False
        else:
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")
        return False    


def docker_clean_config(container_id, src_dir, meta_projects):
    # The list of commands to be executed inside the container.
    commands = [
        # Switch to the source directory.
        f"cd {shlex.quote(src_dir)}",
        # Makefile 
        "make clean || true",
        "make distclean || true",
        "make mrproper || true",
        "make realclean || true",
        "make maintainer-clean || true",
        # Autotools
        "./configure --clean || true",
        "autoreconf --clean || true",
        "autoclean || true",
        # CMake
        "rm -rf CMakeFiles/ || true",
        "rm -f CMakeCache.txt cmake_install.cmake || true",
        "rm -rf build/ || true",
        # Meson
        "rm -rf meson-logs/ meson-private/ || true",
        # cache
        "rm -rf config.cache config.log config.status || true",
        "rm -rf autom4te.cache/ || true",
        # /work
        "rm -rf /work || true ",
        "rm -rf generated/ || true ",
        # f"rm -rf /work/{meta_projects}"
        "mkdir -p /work || true",
        "mkdir -p build/ || true",
        "mkdir -p generated/ || true"
    ]
    command = ' && '.join(commands)
    docker_command = f"docker exec {container_id} sh -c \"{command}\""
    process = subprocess.run(
        docker_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if process.returncode == 0:
        logger.info("Successfully cleaned the configuration and mkdir build/ .")
    else:
        logger.warning(f"Clean command exited with code {process.returncode}")
        logger.warning(f"Output: {process.stdout}")
        logger.warning(f"Errors: {process.stderr}")


    rm_command =  "find /src -type d -name build -exec rm -rf {} + "
    exec_command = f"docker exec {container_id} bash -c \"{rm_command}\" "    
    try:
        subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
        logger.info(f"{rm_command} success")
    except subprocess.CalledProcessError as e:  
        logger.error(f"{rm_command} Error, running ARVO in container:\n{e.output}")

    return process.returncode


def clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path):
    try:
        # stop container_id
        stop_command = f"docker stop {container_id}"
        subprocess.run(stop_command, shell=True)
        
        if os.path.exists(source_labels_path):
            os.makedirs(os.path.dirname(target_labels_path), exist_ok=True)
            os.rename(source_labels_path, target_labels_path)
            logger.info(f"Moved {source_labels_path} to {target_labels_path}")
        else:
            logger.warning(f"clean_docker_env Source file {source_labels_path} does not exist. Skipping move.")
        
        container_remove_command = f"docker stop {container_id}"
        subprocess.run(container_remove_command, shell=True, check=True, text=True)
        logger.info(f"clean_docker_env stop container {container_id}, meta_name:{meta_name}")

        container_remove_command = f"docker rm -f {container_id}"
        subprocess.run(container_remove_command, shell=True, check=True, text=True)
        logger.info(f"clean_docker_env Removed container {container_id}, meta_name:{meta_name}")

        # image_name = f"n132/arvo:{meta_name}-vul"
        # image_remove_command = f"docker rmi -f {image_name}"
        # subprocess.run(image_remove_command, shell=True, check=True, text=True)
        # logger.info(f"Removed image {image_name}")    
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}\n{e.output}")
    except Exception as e:
        logger.error(f"container_id:{container_id} An error occurred: {e}")


# Automatically optimize compilation commands by adding necessary escape characters 
# for file names referenced in macro definitions.
# e.g: 15374 -DMBEDTLS_CONFIG_FILE="\"mbedtls-config.h\""

def is_header_file(value):
    value = value.strip('"') 
    return value.endswith(('.h', '.hpp'))

def optimize_compile_command(command):
    macro_pattern = re.compile(r'-D([A-Za-z0-9_]+)=(".*?"|\'.*?\'|\S+)')

    def escape_quotes(value):
        if value.startswith('"') and value.endswith('"'):
            logger.info(f"optimize_compile_command-1 value : {value} ")
            if is_header_file(value):
                inner = value[1:-1].replace('"', '\\"')
                return f'"\\"{inner}\\""'
            else:
                return value
        elif value.startswith('\\\"') and value.endswith('\\\"'):
            logger.info(f"optimize_compile_command-2 value : {value} ")
            # inner = value[1:-1].replace('\\\"', '\\"')
            return f'"{value}"'
        elif value.startswith("'") and value.endswith("'"):
            logger.info(f"optimize_compile_command-3t value : {value} ")
            inner = value[1:-1].replace('"', '\\"').replace("'", "\\'")
            return f'"\\"{inner}\\""'
        else:
            return value
            # return f'"\\"{value}\\""'

    def replace_macro(match):
        macro_name = match.group(1)
        macro_value = match.group(2)
        if re.search(r'\s|["\'(){};]', macro_value):
            escaped_value = escape_quotes(macro_value)
            return f'-D{macro_name}={escaped_value}'
        else:
            return match.group(0)

    optimized_command = macro_pattern.sub(replace_macro, command)
    return optimized_command


def copy_docker_compile_file(container_id, src_dir, compile_command, share_file_folder):

    logger.info(f"copy_docker_compile_file start")
    command_parts = compile_command.split()
    source_files = []
    output_file = ""

    for part in command_parts:
        if part.endswith(".c") or part.endswith(".cpp") or part.endswith(".cc") or part.endswith(".cxx"):
            source_files.append(part)
    logger.info(f"source_files:{source_files}")    

    if "-o" in command_parts:
        index = command_parts.index("-o")
        if index + 1 < len(command_parts):
            output_file = command_parts[index + 1]
    logger.info(f"output_file:{output_file}")   


    for source_file in source_files:
        copy_command = f"cd {src_dir} &&  cp {source_file} {share_file_folder}"
        exec_command = f"docker exec {container_id} bash -c \"{copy_command}\" "     
        
        try:
            subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
            logger.info(f"{copy_command} success")
        except subprocess.CalledProcessError as e:  
            logger.error(f"{copy_command} Error, running ARVO in container:\n{e.output}")

    if output_file:
        copy_command = f"cd {src_dir} &&  cp {output_file} {share_file_folder}"
        exec_command = f"docker exec {container_id} bash -c \"{copy_command}\"  "
        try:
            subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
            logger.info(f"{copy_command} success")
        except subprocess.CalledProcessError as e:  
            logger.error(f"{copy_command} Error, running ARVO in container:\n{e.output}")

def extract_patch_files(patch_json_file):
    if not os.path.exists(patch_json_file):
        print(f"patch_json_file non-existent: {patch_json_file}")
        return []
    try:
        with open(patch_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        patch_files = data.get('patch_files', [])
        return patch_files
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file :{patch_json_file}, error: {e}")
        return []
    

def run_docker_command(container_id, command, timeout=600, check_interval=10):
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output_queue = queue.Queue()

        def enqueue_output(out, queue):
            for line in iter(out.readline, ''):
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                logger.info(f"{clean_line.strip()} \n ")
                queue.put(time.time())  
            out.close()

        stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, output_queue))
        stdout_thread.daemon = True
        stdout_thread.start()
        last_output_time = time.time()
        # check for timeout
        while True:
            try:
                output_time = output_queue.get(timeout=check_interval)
                last_output_time = output_time
            except queue.Empty:
                pass 
            if process.poll() is not None:
                break
            if time.time() - last_output_time > timeout:
                logger.error(f"No output for {timeout} seconds. Terminating the command...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break

        stdout_thread.join(timeout=5)
        if stdout_thread.is_alive():
            logger.warning("Output thread did not finish in time.")
            return -1
        stderr_output = ''
        try:
            stderr_output = process.stderr.read()
            if stderr_output:
                logger.error(f"Error during command execution:\n{stderr_output}")
        except Exception as e:
            logger.error(f"Error reading stderr: {e}")

        returncode = process.returncode if process.returncode is not None else -1
        return returncode

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return -1


#  compile_commands.json ，Extract the compilation command items that match patch_files.
def extract_bear_compile_command(compile_command_file, patch_files):
    file_name_pattern = re.compile(r"[^/\\]+(?=\.[a-zA-Z0-9]+$)")  # 匹配文件名
    patch_filenames = {re.search(file_name_pattern, patch_file).group(0) for patch_file in patch_files }

    for filename in patch_filenames:
        logger.info(f"extract_bear_compile_command patch_files: {filename}")

    match_compile_commands = []
    try:
        with open(compile_command_file, 'r') as f:
            compile_commands = json.load(f)
        for entry in compile_commands:
            file_path = entry.get("file", "")
            match = re.search(file_name_pattern, file_path)
            filename = match.group(0)
            # logger.info(f"extract_bear_compile_command compile_commands file: {filename}")
            if filename in patch_filenames:
                match_compile_commands.append(entry)
        return match_compile_commands
    
    except FileNotFoundError:
        logger.error(f"Warning: {compile_command_file} file cannot be found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Warning: {compile_command_file}  Not a valid JSON file.")
        return None
    

def Generat_docker_compile_command(patch_sourcefiles_compile_command, is_clang_12, is_delete_sanitize, bear_version, arvo_clang_version):
    output_filename_suffix=".ll"

    # Parse bear format
    if bear_version == "2.4.3":
        logger.info(f"Generat_docker_compile_command bear version: {bear_version}")
        command_parts = patch_sourcefiles_compile_command.get("arguments", [])
    elif bear_version == "2.1.5":
        logger.info(f"Generat_docker_compile_command bear version: {bear_version}")
        command_parts = patch_sourcefiles_compile_command.get("command", "").split()
    else:
        logger.error(f"Unsupported bear version: {bear_version}")
        return None

    if not command_parts:
        logger.error("The command parameter list is empty")
        return None

    # Process the compilation options
    modified_command_parts = []
    
    skip_next = False  # Mark whether to skip the next argument (e.g., the output file after '-o')".
    
    # Regular expression pattern
    afl_related_options = [
        r'^-fpass-plugin=.*$',  # -fpass-plugin=/src/aflplusplus/afl-llvm-pass.so
        r'^-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION.*$',  # -DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION=1
        r'^-D__AFL_.*$',  # -D__AFL_INIT()=...
        r'__afl_.*',   # delete expressions starting with _afl"
        r'.*SIG_AFL_.*',   # 
        r'^-fsanitize=fuzzer.*$',       
        r'^-fsanitize-coverage=.*$',    
        r'^-fno-omit-frame-pointer$',   
        r'.*afl-llvm-pass.so.*',       
    ]

    sanitizer_related_options = [
        "-fsanitize=address",
        "-fsanitize-address-use-after-scope",
        r'^-fsanitize=.*$',             # Delete all options related to -fsanitize
    ]


    # compilation options that need to be added
    additional_options = ["-O0", "-gline-tables-only", "-emit-llvm", "-S"]

    # Precompiled regular expression
    afl_patterns = [re.compile(pattern) for pattern in afl_related_options]
    if is_delete_sanitize: 
        sanitizer_patterns = [re.compile(pattern) for pattern in sanitizer_related_options]
    
    # compiler settings
    if is_clang_12:
        command_parts[0] = "clang-12"
    else:
        command_parts[0] = "clang"
        # major_clang_match = re.match(r"(\d+)\.\d+\.\d+", arvo_clang_version)
        # major_version = major_clang_match.group(1)  
        # clang_compiler = f"clang-{major_version}"
        # command_parts[0] = clang_compiler
    
    
    for i, part in enumerate(command_parts):
        if skip_next:
            skip_next = False
            continue

        # Process -o and modify the output file name
        if part == "-o":
            modified_command_parts.append(part)
            if i + 1 < len(command_parts):
                output_file = command_parts[i + 1]
                base_output = os.path.splitext(output_file)[0]
                new_output = base_output + output_filename_suffix
                
                modified_command_parts.append(new_output)
                skip_next = True  
            else:
                logger.error("Missing output file name after '-o' option")
                return None
            continue

        # Delete AFL related compilation options:
        if any(pattern.match(part) for pattern in afl_patterns):
            logger.debug(f"Delete AFL related compilation options: {part}")
            continue


        if is_delete_sanitize and any(pattern.match(part) for pattern in sanitizer_patterns):
            logger.debug(f"Delete compiler options related to sanitizer: {part}")
            continue

        # Remove optimization options -O* 和 -Ofast
        if re.match(r'^-O\d$', part) or part.startswith('-Ofast'):
            logger.debug(f"Delete optimization options: {part}")
            continue

  
        if part == "-c":
            logger.debug(" Delete the '-c' option ")
            continue


        if part == "-Xclang":
            logger.debug("Delete the '-Xclang' option")
            continue
        

        # macro definitions with spaces 
        if part.startswith('-D') and '=' in part:
            define, value = part[2:].split('=', 1)
            if ' ' in value:
                value = f'"{value}"'
            processed_part = f'-D{define}={value}'
            modified_command_parts.append(processed_part)
            logger.debug(f"Processed macro definition: {processed_part}")
            continue
        modified_command_parts.append(part)

    # Add compilation options"
    modified_command_parts += additional_options
    logger.debug(f"Add compilation options: {additional_options}")

    modified_command = " ".join(modified_command_parts)
    logger.debug(f"Modified compilation command: {modified_command}")

    return modified_command


def Generat_docker_compile_commands(patch_sourcefiles_compile_commands, is_clang_12, is_delete_sanitize, bear_version, arvo_clang_version):
    
    docker_compile_commands = []
    for compile_command in patch_sourcefiles_compile_commands:
        docker_compile_command = Generat_docker_compile_command(compile_command, is_clang_12, is_delete_sanitize, bear_version, arvo_clang_version)
        logger.info(f"Generat_docker_compile_commands, docker_compile_command:{docker_compile_command} ")
        if docker_compile_command:
            docker_compile_commands.append({
                "docker_compile_command": docker_compile_command,
                "directory": compile_command.get("directory", "")
            })
        else:
            logger.warning(f"Failed to generate Generat_docker_compile_command, compile_command:{compile_command} ")
    return docker_compile_commands


# Execute each compilation command within the Docker container
def run_docker_compile_command(container_id, docker_compile_commands, IR_Docker_CollectData_Folder, is_clang_12, is_delete_sanitize):
    CollectData_flag = False
    for cmd_entry in docker_compile_commands:
        docker_compile_command = cmd_entry["docker_compile_command"]
        optimize_docker_compile_command = optimize_compile_command(docker_compile_command)
        src_directory = cmd_entry["directory"]
        logger.info(f"Executing command: {docker_compile_command} in directory: {src_directory}")
        docker_compile_command = f"cd {src_directory} && {optimize_docker_compile_command}"
        run_compile_command = f"docker exec -it {container_id}  bash -c \'{docker_compile_command}\' "
        return_code = run_docker_command(container_id, run_compile_command)
        if return_code == 0:
            copy_docker_compile_file(container_id, src_directory, docker_compile_command, IR_Docker_CollectData_Folder)
            CollectData_flag = True
        else:
            logger.error(f"run_docker_compile_command failed, container_id:{container_id} , docker_compile_command:{docker_compile_command}")
    return CollectData_flag

def Updata_docker_sources_list(container_id):
    # docker: copy /etc/apt/sources.list
    copy_command = f"cp /ShareFiles/ShareFiles/sources.list /etc/apt/sources.list"
    exec_command = f"docker exec {container_id} {copy_command}"
    try:
        subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
        logger.info(f"copy /etc/apt/sources.list success")
    except subprocess.CalledProcessError as e:  
        logger.error(f"Error while running ARVO in container:\n{e.output}")

def Install_docker_bear(container_id):
    # docker: apt update   
    update_command = f"docker exec -it {container_id} apt-get  update"
    returncode = run_docker_command(container_id, update_command)
    if returncode == 0:
        logger.info("APT sources updated successfully!")
    else:
        logger.error(f"APT update failed with return code {returncode}")

    # docker: install soft 
    update_command = f"docker exec  {container_id} apt install -y bear"
    returncode = run_docker_command(container_id, update_command)
    if returncode == 0:
        logger.info("apt install  origin bear successfully!")
    else:
        logger.error(f"apt install failed with return code {returncode}")
        return

    # docker: bear --version
    arvo_command = "bear --version"
    bear_version = "" 
    exec_command = f"docker exec {container_id} {arvo_command}"
    try:
        arvo_output = subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
        bear_version_match = re.search(r"bear (\S+)", arvo_output)
        if bear_version_match:
            bear_version = bear_version_match.group(1) 
            logger.info(f"bear version: {bear_version}")
        else:
            logger.info("bear version not found in the output.")
    except subprocess.CalledProcessError as e:
        logger.error(f"bear version fail")

    if not bear_version:
        # docker: copy /etc/apt/sources.list
        Updata_docker_sources_list(container_id)
        # docker: apt update   
        update_command = f"docker exec -it {container_id} apt-get  update"
        returncode = run_docker_command(container_id, update_command)
        if returncode == 0:
            logger.info("APT sources updated successfully!")
        else:
            logger.error(f"APT update failed with return code {returncode}")

        # docker: install soft 
        update_command = f"docker exec  {container_id} apt install -y bear"
        returncode = run_docker_command(container_id, update_command)
        if returncode == 0:
            logger.info("apt install bear-2.4 successfully!")
        else:
            logger.error(f"apt install failed with return code {returncode}")
            return
            # docker: bear --version
        arvo_command = "bear --version"
        exec_command = f"docker exec {container_id} {arvo_command}"
        try:
            arvo_output = subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
            bear_version_match = re.search(r"bear (\S+)", arvo_output)
            if bear_version_match:
                bear_version = bear_version_match.group(1)
                logger.info(f"bear version: {bear_version}")
            else:
                logger.info("Clang version not found in the output.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing command: {e}")

    return bear_version

# step 2: bear Parse command-line arguments  and clang-12 compile
def process_labels_json_file(meta_labels_file, config):
    # logger.info(f"******************************debug******************************")

    logger.info(f"process_labels_json_file : {meta_labels_file}")

    meta_folder = os.path.abspath(config["meta_folder"])
    meta_name = meta_labels_file.split('.')[0]
    project_type = config["project_type"]

    logger.info(f"*****************download  Docker *****************")
    json_file_fullpath = os.path.join(meta_folder, f"{meta_name}.json")

    if not os.path.isfile(json_file_fullpath):
        logger.info(f"File does not exist: {json_file_fullpath}")
        return
    try:
        with open(json_file_fullpath, 'r') as f:
            data = json.load(f)
            if "project" in data:
                meta_projects = data["project"]
                logger.info(f"meta_projects:{meta_projects}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return   

    #  Pulling Docker n132/arvo:{meta_name}-vul
    image_name = f"n132/arvo:{meta_name}-{project_type}"
    logger.info(f"Pulling Docker image: {image_name}...")
    pull_command = f"docker pull {image_name}"

    # with subprocess.Popen(pull_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
    #     for line in proc.stdout:
    #         logger.info(f"{line.strip()}")
    #     for line in proc.stderr:
    #         logger.error(f"Failed:{line.strip()}")
    #         return
    # if proc.returncode == 0:
    #     logger.info(f"Image {image_name} pulled successfully: {image_name}")
    # else:
    #     logger.error(f"Failed to pull image {image_name}")
    #     return

    CollectData_Folder = config["CollectData_Folder"]
    share_file_folder = config["share_file_folder"]
    share_file_folder = os.path.expanduser(f"{share_file_folder}")
    logger.info(f"Image: {image_name}, share_file_folder: {share_file_folder}")
    run_command = f"docker run -d -v {share_file_folder}:/ShareFiles {image_name} sleep infinity"
    

    logger.info(f"Running Docker container: {run_command}")
    container_result = subprocess.run(run_command, shell=True, capture_output=True, text=True)
    if container_result.returncode != 0:
        logger.error(f"Failed to start container: {container_result.stderr}")
    container_id = container_result.stdout.strip()



    logger.info(f"{image_name} Container {container_id} started.")
    
    # current_user = check_docker_user(container_id)
    # logger.info(f"Container {container_id} ,current_user:{current_user}")
    
    # docker: pwd
    arvo_command = "pwd" 
    exec_command = f"docker exec {container_id} {arvo_command}"
    logger.info(f"Running arvo in container: {arvo_command}")
    try:
        arvo_output = subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
        src_dir = arvo_output.strip()
        # if src_dir == "/src":
        #     src_dir = f"/src/{meta_projects}"
        logger.info(f"ARVO Output: {arvo_output.strip()}, src_dir:{src_dir}")
    except subprocess.CalledProcessError as e: 
        logger.error(f"Error while running ARVO in container:\n{e.output}")

    # docker: clang -v
    arvo_command = "clang -v"
    arvo_clang_version = "" 
    exec_command = f"docker exec {container_id} {arvo_command}"
    try:
        arvo_output = subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
        version_match = re.search(r"clang version (\S+)", arvo_output)
        if version_match:
            arvo_clang_version = version_match.group(1) 
            logger.info(f"Clang version: {arvo_clang_version}")
        else:
            logger.info("Clang version not found in the output.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {e}")

    # Updata_docker_sources_list(container_id)
    bear_version = Install_docker_bear(container_id)

    projects_outpath = f"{share_file_folder}{CollectData_Folder}/{meta_name}/{project_type}"
    logger.info(f"projects_outpath: {projects_outpath}")
    os.makedirs(projects_outpath, exist_ok=True) 
    docker_env_outfilepath = f"{projects_outpath}/docker_env_file.json"
    docker_info = {
        "image_name": image_name,
        "pwd_dir": src_dir,
        "meta_projects": meta_projects,
        "arvo_clang_version": arvo_clang_version,
        "bear_version": bear_version
    }
    with open(docker_env_outfilepath, "w") as json_file:
        json.dump(docker_info, json_file, indent=4)  

    if not bear_version:
        logger.error(f"bear install fail")
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        if meta_projects in G_Trouble_Projects:
            target_labels_path = f"{output_labels_Folder}/really_failed/{meta_name}.json"
        else:
            target_labels_path = f"{output_labels_Folder}/arvo_compile_fail/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        logger.error(f"image_name:{image_name}, bear arvo compile fail")
        return
    

    # docker_clean_config clean
    meta_projects_dir = f"/src/{meta_projects}"
    docker_clean_config(container_id, meta_projects_dir, meta_projects)
    logger.info("docker_clean_config finish!")

    # bear compile
    compile_command = f"cd {src_dir} && bear arvo compile"
    # exec_command = f"docker exec {container_id} proxychains bash -c \"{compile_command}\" "
    exec_command = f"docker exec -it {container_id} bash -c \"{compile_command}\" "
    logger.info("start bear arvo compile, Please wait a few minutes ")
    returncode = run_docker_command(container_id, exec_command)
    if returncode == 0:
        logger.info("bear arvo compile successfully!")    
    else:
        logger.error(f"bear arvo compile failed meta_name:{meta_name}, try to find compile_commands.json")
 
    #     logger.error(f"bear arvo compile failed with return code {returncode}, container_id:{container_id}, meta_name:{meta_name}")
    #     meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
    #     output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
    #     source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
    #     if meta_projects in G_Trouble_Projects:
    #         target_labels_path = f"{output_labels_Folder}/really_failed/{meta_name}.json"
    #     else:
    #         target_labels_path = f"{output_labels_Folder}/arvo_compile_fail/{meta_name}.json"
    #     clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
    #     logger.error(f"image_name:{image_name}, bear arvo compile fail")
    #     return
    

    # bear arvo compile  compile_commands.json
    if( not check_file_in_container(container_id , f"{src_dir}/compile_commands.json") ):
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        if meta_projects in G_Trouble_Projects:
            target_labels_path = f"{output_labels_Folder}/really_failed/{meta_name}.json"
        else:
            target_labels_path = f"{output_labels_Folder}/arvo_compile_fail/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        logger.error(f"image_name:{image_name}, bear arvo compile compile_commands.json non-existent")
        return
    
    
    # save compile_commands.json
    docker_projects_outpath = f"/ShareFiles/{CollectData_Folder}/{meta_name}/{project_type}"
    copy_command = f"cp {src_dir}/compile_commands.json {docker_projects_outpath}/compile_commands.json"
    exec_command = f"docker exec {container_id} {copy_command}"
    try:
        subprocess.check_output(exec_command, shell=True, stderr=subprocess.STDOUT, text=True)
        logger.info(f"bear arvo compile  copy compile_commands.json success")
    except subprocess.CalledProcessError as e:  
        logger.error(f"bear arvo compile  copy compile_commands.json fail ")


    # Retrieve the compilation options related to vulnerabilities
    meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
    meta_labels_file_fullpath = f"{meta_labels_Folder}/{meta_labels_file}"
    patch_sourcefiles = extract_patch_files(meta_labels_file_fullpath)
    Local_compile_command_file = f"{CollectData_Folder}/{meta_name}/{project_type}/compile_commands.json"
    patch_sourcefiles_compile_command = extract_bear_compile_command(Local_compile_command_file, patch_sourcefiles)
    # sace patch_sourcefiles_compile_command
    sourcefiles_compile_file = f"{CollectData_Folder}/{meta_name}/{project_type}/sourcefiles_compile_command.json"
    with open(sourcefiles_compile_file, 'w') as f:
        json.dump(patch_sourcefiles_compile_command, f, indent=4)

    logger.info(f"finish extract_bear_compile_command")
    ##################################compile original clang #########################################################################


    if not patch_sourcefiles_compile_command:
        logger.error(f"image_name:{image_name}, patch_sourcefiles_compile_command is None")
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        target_labels_path = f"{output_labels_Folder}/arvo_compile_fail/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        return


    # retain sanitize
    is_clang_12 = False
    is_delete_sanitize = False
    IR_Local_CollectData_Folder = f"{share_file_folder}{CollectData_Folder}/{meta_name}/{project_type}/clang/retain_sanitize"
    os.makedirs(IR_Local_CollectData_Folder, exist_ok=True) 
    IR_Docker_CollectData_Folder = f"{docker_projects_outpath}/clang/retain_sanitize"

    docker_compile_commands = Generat_docker_compile_commands(patch_sourcefiles_compile_command, is_clang_12, is_delete_sanitize, bear_version, arvo_clang_version)
    compile_res = run_docker_compile_command(container_id, docker_compile_commands, IR_Docker_CollectData_Folder, is_clang_12, is_delete_sanitize)
    if not compile_res:
        logger.error(f"image_name:{image_name}, original retain sanitize error, docker_compile_commands:{docker_compile_commands}")
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        target_labels_path = f"{output_labels_Folder}/clang_retain_sanitize_fail/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        return
    

    # delete sanitize
    is_clang_12 = False
    is_delete_sanitize = True
    IR_Local_CollectData_Folder = f"{share_file_folder}{CollectData_Folder}/{meta_name}/{project_type}/clang/delete_sanitize"
    os.makedirs(IR_Local_CollectData_Folder, exist_ok=True) 
    logger.info(f"is_clang_12:{is_clang_12}, is_delete_sanitize:{is_delete_sanitize} start")
    IR_Docker_CollectData_Folder = f"{docker_projects_outpath}/clang/delete_sanitize"
    docker_compile_commands = Generat_docker_compile_commands(patch_sourcefiles_compile_command, is_clang_12, is_delete_sanitize, bear_version, arvo_clang_version)
    if not docker_compile_commands:
        logger.error(f"image_name:{image_name}, original delete sanitize, docker_compile_commands is NONE")
    compile_res = run_docker_compile_command(container_id, docker_compile_commands, IR_Docker_CollectData_Folder, is_clang_12, is_delete_sanitize)
    if not compile_res:
        logger.error(f"image_name:{image_name}, original delete sanitize, docker_compile_commands:{docker_compile_commands}")
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        target_labels_path = f"{output_labels_Folder}/clang_delete_sanitize_fail/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        return
    

    ##################################compile clang 12 #########################################################################
    if re.match(r"^12\.0(\.\d+)*$", arvo_clang_version):
        logger.info(f"arvo_clang:{arvo_clang_version} is 12.0.x version.")
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        target_labels_path = f"{output_labels_Folder}/success/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        return

    Updata_docker_sources_list(container_id)
    # docker: apt update   
    update_command = f"docker exec {container_id} apt-get  update"
    returncode = run_docker_command(container_id, update_command)
    if returncode == 0:
        logger.info("APT sources updated successfully!")
    else:
        logger.error(f"APT update failed with return code {returncode}")

    # install clang-12
    update_command = f"docker exec {container_id} apt install -y clang-12 llvm-12 file bash "
    returncode = run_docker_command(container_id, update_command)
    if returncode == 0:
        logger.info("apt install clang-12 successfully!")
    else:
        logger.error(f"image_name:{image_name}, install clang-12 fail")
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        target_labels_path = f"{output_labels_Folder}/clang12_fail/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        return
    

    # retain sanitize
    is_clang_12 = True
    is_delete_sanitize = False
    IR_Local_CollectData_Folder = f"{share_file_folder}{CollectData_Folder}/{meta_name}/{project_type}/clang12/retain_sanitize"
    os.makedirs(IR_Local_CollectData_Folder, exist_ok=True) 
    IR_Docker_CollectData_Folder = f"{docker_projects_outpath}/clang12/retain_sanitize"
    docker_compile_commands = Generat_docker_compile_commands(patch_sourcefiles_compile_command, is_clang_12, is_delete_sanitize, bear_version, arvo_clang_version)
    compile_res = run_docker_compile_command(container_id, docker_compile_commands, IR_Docker_CollectData_Folder, is_clang_12, is_delete_sanitize)
    if not compile_res:
        logger.error(f"image_name:{image_name}, clang12 retain sanitize, docker_compile_commands:{docker_compile_commands}")


    # delete sanitize   
    is_clang_12 = True
    is_delete_sanitize = True
    IR_Local_CollectData_Folder = f"{share_file_folder}{CollectData_Folder}/{meta_name}/{project_type}/clang12/delete_sanitize"
    os.makedirs(IR_Local_CollectData_Folder, exist_ok=True) 
    logger.info(f"is_clang_12:{is_clang_12}, is_delete_sanitize:{is_delete_sanitize} start")
    
    IR_Docker_CollectData_Folder = f"{docker_projects_outpath}/clang12/delete_sanitize"
    docker_compile_commands = Generat_docker_compile_commands(patch_sourcefiles_compile_command, is_clang_12, is_delete_sanitize, bear_version, arvo_clang_version)
    compile_res = run_docker_compile_command(container_id, docker_compile_commands, IR_Docker_CollectData_Folder, is_clang_12, is_delete_sanitize)
    if not compile_res:
        logger.error(f"image_name:{image_name}, original delete sanitize, docker_compile_commands:{docker_compile_commands}")
        meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
        output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
        source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
        target_labels_path = f"{output_labels_Folder}/clang12_delete_sanitize_fail/{meta_name}.json"
        clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)
        return


    logger.info(f"finish, image_name:{image_name}, docker_compile_commands success ")
    meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
    output_labels_Folder = os.path.abspath(config["Result_Labels_Save_Folder"])
    source_labels_path = f"{meta_labels_Folder}/{meta_name}.json"
    target_labels_path = f"{output_labels_Folder}/success/{meta_name}.json"
    clean_docker_env(container_id, meta_name, source_labels_path, target_labels_path)




if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ["yes", "true", "t", "1", "y"])
    args = parser.parse_args()
    add_args(parser)
    args = parser.parse_args()
    logger.info(f" args.config: {args.config} ")
    config = load_config(args.config)

    # step 1: Analyze patch files and extract function names
    logger.info(f" start step 1: Analyze patch files and extract function names........ ")
    Patches_Folder = os.path.abspath(config["Patches_Folder"])
    Patches_diff_files = [f for f in os.listdir(Patches_Folder) if f.endswith('.diff')]
    num_processes = 16
    
    process_with_diff_files = partial(process_Patches_diff_file, config=config)
    with Pool(processes=num_processes) as pool:
        pool.map(process_with_diff_files, Patches_diff_files)

    logger.info(f" finish step 1: Analyze patch files and extract function names........ ")

    # sys.exit() 

    # step 2: bear Parse command-line arguments  and clang-12 compile
    logger.info(f" start step 2: bear Parse command-line arguments  and clang-12 compile........ ")
    meta_labels_Folder = os.path.abspath(config["CollectLabels_Folder"])
    meta_labels_files = [f for f in os.listdir(meta_labels_Folder) if f.endswith('.json')]
    num_processes = 1
    process_with_labels_jsonfile = partial(process_labels_json_file, config=config)
    with Pool(processes=num_processes) as pool:
        pool.map(process_with_labels_jsonfile, meta_labels_files)
    logger.info(f" finish WelkIR_Dataprocess........ ")



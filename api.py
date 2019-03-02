import subprocess
import json
import os
import time
import string
import sys

dataset_path = "datasets/"
script_path = "scripts/"
output_path = "output/"

dataset_meta_path = "datasets/dataset-metadata.json"
kernel_meta_path = "scripts/kernel-metadata.json"


def create_dataset(name):
    '''
    Executes Kaggle API to create new dataset
    PARAMETERS: name - Name of dataset
    '''
    subprocess.call(["kaggle", "datasets", "init", "-p", dataset_path])
    with open(dataset_meta_path, 'r', encoding='utf-8') as metaFile:
        metadata = json.load(metaFile)
    metadata["title"] = name
    metadata["id"] = metadata["id"].replace("INSERT_SLUG_HERE", name)
    with open(dataset_meta_path, "w") as metaFile:
        json.dump(metadata, metaFile)
    subprocess.call(["kaggle", "datasets", "create", "-p", dataset_path])
    print("Dataset pushed")
    return metadata["id"]


def create_kernel(name, dataset_id):
    '''
    Executes Kaggle API to create new kernel
    PARAMETERS:
    name - Name of kernel
    dataset_id - ID of dataset to connect with Kernel
    '''
    name = name + "-kernel"
    subprocess.call(["kaggle", "kernels", "init", "-p", script_path])
    with open(kernel_meta_path, 'r', encoding='utf-8') as metaFile:
        metadata = json.load(metaFile)

    py_files = [f for f in os.listdir(script_path) if f.endswith('.py')]
    metadata["title"] = name
    metadata["id"] = metadata["id"].replace("INSERT_KERNEL_SLUG_HERE", name)
    metadata["code_file"] = py_files[0]
    metadata["language"] = "python"
    metadata["kernel_type"] = "script"
    metadata["dataset_sources"] = [dataset_id]
    with open(kernel_meta_path, "w") as metaFile:
        json.dump(metadata, metaFile)
    subprocess.call(["kaggle", "kernels", "push", "-p", script_path])
    return metadata["id"]


def get_output(name):
    '''
    Executes Kaggle API to get output of kernel
    PARAMETERS:
    name - Name of kernel to execute
    '''
    subprocess.call(["kaggle", "kernels", "output", name, "-p", output_path])


def validate_name(name):
    '''
    Validates kernel/dataset name to NOT include special-characters and spaces.
    Can include (hyphen) "-"
    PARAMETERS:
    name - Kernel/dataset name to validate for
    '''
    invalidChars = set(string.punctuation)
    invalidChars.discard("-")   # Allow hyphen
    invalidChars.add(" ")   # Do NOT allow spaces
    if any(char in invalidChars for char in name):
        return False
    else:
        return True


if __name__ == '__main__':
    name = input("Enter a name to dataset/kernel: ")
    if(validate_name(name) is False):
        sys.exit("Name should not contain space/special-characters, use '-' instead")
    else:
        print("The time of code execution BEGIN is : ", end="")
        print(time.ctime())
        
        dataset_id = create_dataset(name)
        print (dataset_id)
        
        time.sleep(10)
        kernel_id = create_kernel(name, dataset_id)
        print(kernel_id)
        
        time.sleep(10)
        get_output(kernel_id)
        
        print("The time of code execution END is : ", end="")
        print(time.ctime())

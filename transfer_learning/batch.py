import os
import time
import subprocess
import shlex
from itertools import product

import argparse

parser = argparse.ArgumentParser(description='Trojan Detector for Question & Answering Tasks.')

parser.add_argument('--gpu', nargs='+', required=True, type=int, help='Which GPU', )

args = parser.parse_args()

gpu_list = args.gpu
gpus_per_command = 1
polling_delay_seconds = .1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

freeze_levels = ['4']

from constants import DATASET_NAME_TO_PATH
datasets = list(DATASET_NAME_TO_PATH.keys())

from tools import get_supported_model_ids
model_ids = get_supported_model_ids()

import pandas as pd
params = pd.read_csv('parameters.csv')

commands_to_run = [f'python main.py '\
    f'--freeze-level {fl} '\
    f'--model_id {model} '\
    f'--dataset {ds} '\
    f'--epochs 150 '\
    f'--weight-decay 0.0005 '\
    f'--batch-size 64 '\
    f'--step-lr 50 '\
    f'--out-dir results '\
    f'--adv-train 0 '\
    f'--arch resnet50 '\
    f'--lr {params[(params["dataset"]==ds) & (params["freeze_level"]==int(fl))].lr.item()} '\
    for model, ds, fl in product(model_ids, datasets, freeze_levels)]

commands_to_run.reverse()
def poll_process(process):
    time.sleep(polling_delay_seconds)
    return process.poll()

pid_to_process_and_gpus = {}
free_gpus = set(gpu_list)
while len(commands_to_run) > 0:
    time.sleep(polling_delay_seconds)
    # try kicking off process if we have free gpus
    print(f'free_gpus: {free_gpus}')
    while len(free_gpus) >= gpus_per_command:
        print(f'free_gpus: {free_gpus}')
        gpus = []
        for i in range(gpus_per_command):
            # updates free_gpus
            gpus.append(str(free_gpus.pop()))
        command = commands_to_run.pop()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
        subproc = subprocess.Popen(shlex.split(command))
        # updates_dict
        pid_to_process_and_gpus[subproc.pid] = (subproc, gpus)
    
    # update free_gpus
    for pid, (current_process, gpus) in pid_to_process_and_gpus.copy().items():
        if poll_process(current_process) is not None:
            print(f'done with {pid}')
            free_gpus.update(gpus)
            del pid_to_process_and_gpus[pid]
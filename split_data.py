#!/usr/bin/env python3

import shutil
import os

def get_testing_data():
    file_name = 'input/testing/testing_list.txt'
    with open(file_name, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].rstrip()
        src = 'input/train/audio/' + lines[i]
        dest = 'input/testing/audio/' + lines[i]
        dest_dir = 'input/testing/audio/' + lines[i].split('/')[0]
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(src=src, dst=dest)

def get_validation_data():
    file_name = 'input/validation/validation_list.txt'
    with open(file_name, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].rstrip()
        src = 'input/train/audio/' + lines[i]
        dest = 'input/validation/audio/' + lines[i]
        dest_dir = 'input/validation/audio/' + lines[i].split('/')[0]
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(src=src, dst=dest)


if __name__ == '__main__':
    get_testing_data()
    get_validation_data()
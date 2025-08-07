import argparse
import csv
import sys
from collections import deque
from decimal import Decimal
from typing import List

import numpy as np
import regex as re

import pandas as pd
import os
import logging

from dag import TaskDAG, TaskInstance, generate_task_args, JobDAG
from scheduler_job import Job
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def config_cores_memory(options: argparse.Namespace):
    directory = os.path.join(options.directory, "config_sampled")
    target_memory = float(options.memory)
    target_probability = Decimal(options.probability)
    count = 0
    cpu_sum = 0
    mem_sum = 0
    for config_directory in os.listdir(directory):
        probability = Decimal(config_directory.split("_")[-2])
        if target_probability == probability and config_directory.startswith("dag_updated_vms"):
            for file in os.listdir(os.path.join(directory, config_directory)):
                if file.endswith(".csv"):
                    memory = float(file[:-len(".csv")].split("_")[-1])
                    if memory == target_memory:
                        count += 1
                        df = pd.read_csv(os.path.join(directory, config_directory, file))
                        cpu_sum += float(df["CPU"].sum())
                        mem_sum += float(df["Memory"].sum())
    print(f"avg cpu for config mem {target_memory}: {cpu_sum/count}")
    print(f"avg mem for config mem {target_memory}: {mem_sum/count}")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", help="Trace directory")
    parser.add_argument("--operation", "-o", help="What test to run.", default="config")
    parser.add_argument("--memory", "-m", help="Memory percentile")
    parser.add_argument("--probability", "-p", help="Probability")



    options = parser.parse_args(args)

    if options.operation == "config":
        config_cores_memory(options)

if __name__ == "__main__":
    main()

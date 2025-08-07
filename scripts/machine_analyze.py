import os
import argparse
from importlib import import_module
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import pyarrow as pa
from pyarrow import csv
import glob
import sys
from timeit import default_timer as timer

import sys

utilization_import = import_module("0-utilization")
create_plot = getattr(utilization_import, "create_plot_one_per")

def count_machines(trace_file: str):
    columns = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    data = pd.read_csv(trace_file, header=None, names=columns)
    timestamps = []
    current_core = []
    current_memory = []

    data_ndarray = data.to_numpy()

    unique_machines = set()
    for row in data_ndarray:
        machine_id = row[0]
        if machine_id in unique_machines:
            continue
        else:
            unique_machines.add(machine_id)

    print(len(unique_machines))

def max_timestamp(trace_file: str) -> None:
    columns = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    data = pd.read_csv(trace_file, header=None, names=columns)
    print(data.loc[data["time_stamp"].idxmax()])

def graph(options: argparse.Namespace) -> None:
    trace_file = options.trace_file or "../alibaba-trace/machine_usage.csv"
    start = timer()
    columns = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    data = pd.read_csv(trace_file, header=None, names=columns)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    data.sort_values("time_stamp", inplace=True)
    timestamps = []
    current_core = []
    current_memory = []
    core_count = 0
    mem_count = 0
    end = timer()
    print(f"preprocessing: {end - start}")

    start = timer()
    data_ndarray = data.to_numpy()
    del data
    recording_time_stamp = data_ndarray[0][1]
    num_machines = 4023
    end = timer()
    print(f"numpy conversion: {end - start}")

    start = timer()
    for row in data_ndarray:
        # machine_id = row[0]
        time_stamp = int(row[1])
        cpu_util_percent = float(row[2])
        mem_util_percent = float(row[3])
        # mem_gps = row[4]
        # mkpi = row[5]
        # net_in = row[6]
        # net_out = row[7]
        # disk_io_percent = row[8]
        if time_stamp == recording_time_stamp:
            core_count += cpu_util_percent
            mem_count += mem_util_percent
        else:
            timestamps.append(recording_time_stamp)
            current_core.append(core_count / num_machines / 100)  # normalize cpu to total of cluster, to 1
            current_memory.append(mem_count / num_machines / 100)  # normalize mem to total of cluster, to 1
            core_count = cpu_util_percent
            mem_count = mem_util_percent
            recording_time_stamp = time_stamp

            if (options.start is not None and recording_time_stamp < int(options.start)) \
                    or (options.end is not None and recording_time_stamp > int(options.end)):
                timestamps.pop()
                current_core.pop()
                current_memory.pop()
    if core_count > 0 or mem_count > 0:
        timestamps.append(recording_time_stamp)
        current_core.append(core_count / num_machines / 100)
        current_memory.append(mem_count / num_machines / 100)

        if (options.start is not None and recording_time_stamp < int(options.start)) \
                or (options.end is not None and recording_time_stamp > int(options.end)):
            timestamps.pop()
            current_core.pop()
            current_memory.pop()
    del data_ndarray
    end = timer()
    print(f"done with loop, took {end - start}")

    # data[data["time_stamp"].isin([time_stamp])].sum
    # print(data.groupby(['time_stamp'])['cpu_util_percent'].sum() * 96) # multiply by 96 cores for each machine to get cpu cores at this timestamp
    create_plot(timestamps, current_core, current_memory, input_file_name=os.path.basename(trace_file), output_directory=output_dir, write_config=False,
                cores_label="CPU usage (total)", mem_label="Mem usage (total)", output=options.output)

def hist(trace_file: str) -> None:
    columns = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    data = pd.read_csv(trace_file, header=None, names=columns)
    timestamps = data["time_stamp"]
    ax = timestamps.value_counts().plot.bar(x="timestamp", y="num of occurrences")
    fig = ax.get_figure()
    # pdf is 2 mb and does not load on my computer, probably too big
    fig.savefig('histogram.png')

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", "-t", help="Where the machine_usage trace is", default="../alibaba-trace/machine_usage.csv")
    parser.add_argument("--operation", "-o", help="The operation to run on the trace")
    parser.add_argument("--output")
    parser.add_argument("--start")
    parser.add_argument("--end")
    options = parser.parse_args(args)

    trace_file = options.trace_file
    operation = options.operation

    if operation == "count_machines":
        count_machines(trace_file)
    elif operation == "graph":
        graph(options)
    elif operation == "max_timestamp":
        max_timestamp(trace_file)
    elif operation == "hist":
        hist(trace_file)


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import datetime as dt

from importlib import import_module
from typing import List

import numpy as np
import pandas as pd

import time

import logging

from matplotlib import pyplot as plt
from pytz import timezone
import pyarrow as pa
import pyarrow.csv
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

utilization_import = import_module("0-utilization")

get_clean_filename = getattr(utilization_import, "get_clean_filename")
get_plot_filename = getattr(utilization_import, "get_plot_filename")
get_store_filename = getattr(utilization_import, "get_store_filename")
plot_percentiles = getattr(utilization_import, "plot_percentiles")

create_plot = getattr(utilization_import, "create_plot")


def calcProcessTime(starttime, cur_iter, max_iter):
    telapsed = time.time() - starttime
    testimated = (telapsed / cur_iter) * (max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).astimezone(timezone("US/Pacific")).strftime("%H:%M:%S")  # in time

    lefttime = testimated - telapsed  # in seconds

    return int(telapsed), int(lefttime), finishtime

def graph_percentile_cpu_mem(options: argparse.Namespace):
    output_directory = options.output
    plot_filename = os.path.join(output_directory)
    data_path = options.trace_file or "Alibaba/trace/updated_vms.csv"

    if os.path.exists(plot_filename) and not options.overwrite:
        print(f"File at {plot_filename} exists!")
        return

    names = "Start Time,End Time,Core,Memory,Instance Num,Lifetime,Job Name,Task Name".split(",")
    chunksize = 1_000_000
    num_lines = 13289039-1
    max_iter = num_lines // chunksize + 1
    start = time.time()
    lines_read = 0
    pa.set_cpu_count(32)

    cores: List[float] = []
    rss: List[float] = []
    timestamps = []

    current_cores: float = 0.0
    current_rss: float = 0.0
    for j, data in enumerate(pd.read_csv(data_path, engine="c", chunksize=chunksize)):
        data['Core'] = pd.to_numeric(data["Core"], errors='coerce')
        data['Memory'] = pd.to_numeric(data["Memory"], errors='coerce')
        data['Memory'] = pd.to_numeric(data['Start Time'], errors='coerce')
        data['Memory'] = pd.to_numeric(data['End Time'], errors='coerce')
        data = data.dropna(subset=["Core", "Memory"])

        # Process the data for utilization calculation
        start_data = data[['Start Time', "Core", "Memory"]].copy()
        start_data.columns = ['timestamp', 'cores', 'rss']
        start_data['event_type'] = 'start'

        end_data = data[['End Time', "Core", "Memory"]].copy()
        end_data.columns = ['timestamp', 'cores', 'rss']
        end_data['event_type'] = 'end'

        combined_data = pd.concat([start_data, end_data]).sort_values('timestamp')
        combined_data_ndarray = combined_data.to_numpy()

        for row in combined_data_ndarray:
            row_timestamp = row[0]
            row_cores = row[1]
            row_rss = row[2]
            row_event_type = row[3]
            if row_event_type == 'start':
                current_cores += row_cores
                current_rss += row_rss
            else:  # event_type is 'end'
                current_cores -= row_cores
                current_rss -= row_rss

            cores.append(current_cores)
            rss.append(current_rss)
            timestamps.append(row_timestamp)
        lines_read += chunksize
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))


    series_core = pd.Series(cores)
    series_mem = pd.Series(rss)
    percents = np.array(list(range(0, 101, 1)))/100
    percentiles_core = series_core.quantile(percents)
    percentiles_mem = series_mem.quantile(percents)

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots()
    color = 'tab:red'

    ax1.set_xlabel("Percentile")
    ax1.set_ylabel("CPU Cores", color=color)
    ax1.plot(percents, percentiles_core, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'

    ax2.set_ylabel('Memory (norm-1)', color=color)  # we already handled the x-label with ax1
    ax2.plot(percents, percentiles_mem, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('CPU/Mem Percentiles')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    end = time.time()
    print(f"Plot graphed in {end - start} seconds")
    print(f"Plot saved to: {plot_filename}")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", "-t", help="Where the machine_usage trace is", default="Alibaba/trace/updated_vms.csv")
    parser.add_argument("--operation", "-o", help="The operation to run on the trace")
    parser.add_argument("--output", help="output directory", default="trace_analysis.png")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--infile", default=None)
    options = parser.parse_args(args)

    graph_percentile_cpu_mem(options)

if __name__ == "__main__":
    main()
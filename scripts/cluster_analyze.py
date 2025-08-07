import argparse
import functools
import glob
import math
import os
import sys
import datetime as dt
from collections import defaultdict
from functools import wraps

from importlib import import_module
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

import time

import logging

from matplotlib import pyplot as plt
from pyarrow import csv

import csv

from pytz import timezone

from scipy.stats import gaussian_kde

import dask.config
from dask.distributed import Client
import dask.dataframe as dd
from functools import wraps

from contextlib import contextmanager

from batch_instance_analyze import calcProcessTime

plot_import = import_module("3-plot")
calculate_cdf = getattr(plot_import, "calculate_cdf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('dask').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)

utilization_import = import_module("0-utilization")


def limit_dask_client_decorator(cpu_cores, mem_gb):
    """
    Decorator factory to limit Dask resource usage using a Dask Distributed Client.

    Returns a decorator that, when applied to a function, will execute that
    function within the context of a Dask Distributed Client configured with
    the specified CPU cores and memory limits.
    """

    def decorator(func):
        """The actual decorator that wraps the function."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function that sets up the Dask Client and calls the original function."""
            memory_limit_str = f'{mem_gb}GB'

            print("--- Debugging inside limit_dask_client_decorator wrapper ---")
            print(f"  Creating Dask Client with: cpu_cores={cpu_cores}, mem_gb={mem_gb}")
            print(f"  Memory limit string used for Client: {memory_limit_str}")
            print("--- End Debugging ---")

            client = Client(n_workers=cpu_cores, threads_per_worker=1, memory_limit=memory_limit_str)  # Explicit Client

            try:
                return func(*args, **kwargs)  # Execute the original function within the Client context
            finally:
                client.close()  # Ensure client is closed after function execution

        return wrapper

    return decorator


def just_requested_utilization(options):
    output_directory = "cluster_analysis"

    requested_utilization = os.path.join(output_directory, "machine_usage_subscribed.csv")
    requested_utilization_port(requested_utilization, options)


def requested_utilization_port(output, options):
    """
    This is copied from batch_instance_analyze::reciprocal_bloat2
    """
    start = time.time()
    data_path = "../alibaba-trace/batch_instance_sort.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    alibaba_trace = pd.read_csv("alibaba-2018-sorted.csv")
    alibaba_trace = alibaba_trace[["job_name", "task_name", "plan_cpu", "plan_mem"]]
    alibaba_trace = alibaba_trace.dropna()
    logger.info("extracting plan_cpu and plan_mem")
    # convert both to an int equivalent representation to avoid slow float operations
    alibaba_data = {f"{row[0]},{row[1]}": (int(row[2]), int(math.ceil(row[3] * 100))) for row in alibaba_trace.to_numpy()}
    del alibaba_trace

    # code copied from extract_batch_instance_type
    chunksize = 10_000_000

    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1
    # requested_data = pd.read_csv(batch_task_request, names=names, engine="c", header=None)
    recording_time_stamp = None
    current_cpu_max = 0.0
    current_cpu_avg = 0.0
    current_mem_max = 0.0
    current_mem_avg = 0.0

    # calculating the requested resources sum per loop is too expensive
    # so maintain a running sum
    running_plan_cpu = 0
    running_plan_mem = 0
    not_found = 0
    total = 0
    end = time.time()
    logger.info(f"preprocessing done in {end - start} seconds")
    start = time.time()

    max_timestamp = options.max_timestamp
    min_timestamp = options.min_timestamp
    early_break = False

    start_chunk = list()
    end_chunk = list()
    chunk = list()
    event_end = 0
    event_start = 0
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if early_break is True and len(chunk) == 0:
                break
            instance_name, task_name, job_name, task_type, status, start_time, end_time, machine_id, seq_no, total_seq_no, cpu_avg, cpu_max, mem_avg, mem_max = row
            start_time = int(start_time)
            end_time = int(end_time)
            # if start_time == 0 or end_time == 0:
            #     logger.info(f"{start_time}, {end_time}")
            if start_time > end_time:
                # malformed data?
                continue
            start_chunk.append((start_time, job_name, task_name, 'start'))
            end_chunk.append((end_time, job_name, task_name, 'end'))
            if (i + 1) % chunksize == 0:
                last_good_timestamp = 0
                for t in start_chunk:
                    if t[0] > last_good_timestamp:
                        last_good_timestamp = t[0]

                chunk = chunk + start_chunk + end_chunk
                chunk.sort(key=lambda x: x[0])

                timestamps = []
                running_plan_cpus = []
                running_plan_mems = []
                chunk_i = 0
                for timestamp, job_name_nested, task_name_nested, event_type in chunk:
                    if timestamp > last_good_timestamp:
                        break
                    # compute the sum of the requested instances (not what is currently utilized)
                    # use a running sum to avoid O(nm) runtime
                    plan_resources = alibaba_data.get(f"{job_name_nested},{task_name_nested}")
                    if plan_resources is None:
                        # missing data
                        continue

                    plan_cpu = plan_resources[0]
                    plan_mem = plan_resources[1]
                    if event_type == 'start':
                        # add to running sums
                        running_plan_cpu += plan_cpu
                        running_plan_mem += plan_mem
                    else:
                        running_plan_cpu -= plan_cpu
                        running_plan_mem -= plan_mem

                    if min_timestamp is not None and timestamp < min_timestamp:
                        continue
                    if max_timestamp is not None and timestamp > max_timestamp:
                        early_break = True
                        break

                    timestamps.append(timestamp)
                    running_plan_cpus.append(running_plan_cpu)
                    running_plan_mems.append(running_plan_mem)
                    chunk_i += 1
                # todo: divide mem by 100
                parsed_df = pd.DataFrame({"time_stamp": timestamps, "cpu_subscribed": running_plan_cpus, "mem_subscribed": running_plan_mems})
                last_row_per_group_df = parsed_df.groupby('time_stamp').tail(1).reset_index(drop=True)
                # logger.info(last_row_per_group_df)
                last_row_per_group_df.sort_values('time_stamp', inplace=True)
                last_row_per_group_df.to_csv(output, index=False, header=not os.path.exists(output), mode="a")
                logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, (i + 1) % chunksize + 1, max_iter))
                if chunk_i < len(chunk) - 1:
                    chunk = chunk[chunk_i:]
                else:
                    chunk = list()
                end_chunk = list()
                start_chunk = list()


def old_utilization_port(output, options: argparse.Namespace):
    trace_file = "../alibaba-trace/machine_usage_sorted.csv"
    columns = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    chunksize = 1_000_000

    timestamps = []
    total_cpu_utils = []
    total_mem_utils = []
    total_cpu_util = 0
    total_mem_util = 0
    recording_time_stamp = None
    num_machines = 4023

    max_timestamp = options.max_timestamp
    min_timestamp = options.min_timestamp
    early_break = False

    for j, data in enumerate(pd.read_csv(trace_file, header=None, names=columns, chunksize=chunksize)):
        if early_break is True:
            break
        data['time_stamp'] = pd.to_numeric(data['time_stamp'], errors='coerce')
        data.sort_values("time_stamp", inplace=True)

        data_ndarray = data.to_numpy()
        del data

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
            if min_timestamp is not None and time_stamp < min_timestamp:
                continue
            if max_timestamp is not None and time_stamp > max_timestamp:
                early_break = True
                break
            if recording_time_stamp is None:
                recording_time_stamp = time_stamp
            if time_stamp == recording_time_stamp:
                total_cpu_util += cpu_util_percent
                total_mem_util += mem_util_percent
            else:
                timestamps.append(recording_time_stamp)
                total_cpu_utils.append(total_cpu_util / num_machines)  # normalize cpu to total of cluster, to 100
                total_mem_utils.append(total_mem_util / num_machines)  # normalize mem to total of cluster, to 100
                total_cpu_util = cpu_util_percent
                total_mem_util = mem_util_percent
                recording_time_stamp = time_stamp
        if total_cpu_util > 0 or total_mem_util > 0:
            timestamps.append(recording_time_stamp)
            total_cpu_utils.append(total_cpu_util / num_machines)
            total_mem_utils.append(total_mem_util / num_machines)

    (pd.DataFrame({"time_stamp": timestamps, "cpu_util_percent": total_cpu_utils, "mem_util_percent": total_mem_utils})
     .to_csv(output))


def just_individual_machine_utilization(options):
    output_dir = "cluster_analysis"
    for machine_id in range(1, 4023 + 1):
        filename = os.path.join(output_dir, f"machine_usage_m_{machine_id}.csv")
        if os.path.exists(filename):
            os.remove(filename)
    individual_machine_utilization(output_dir, options)


def test_machine_existent(options):
    output_dir = "cluster_analysis"
    for machine_id in range(1, 4023 + 1):
        filename = os.path.join(output_dir, f"machine_usage_m_{machine_id}.csv")
        if not os.path.exists(filename):
            logger.warning(f"file at {filename} does not exist")

def cluster_machine_utilization(options: argparse.Namespace):
    output_directory = "cluster_analysis"
    cluster_utilization = os.path.join(output_directory, "machine_usage_all.csv")

    old_utilization_port(cluster_utilization, options)


def individual_machine_utilization(output_directory, options: argparse.Namespace):
    start = time.time()

    def save_group_to_csv_pandas(group_df, output_dir):
        """
        Saves each machine_id group (Pandas DataFrame) to a separate CSV file.
        Designed for Pandas chunked processing, appending to existing files.
        """
        if group_df.empty:
            logger.warning("save_group_to_csv_pandas received an empty DataFrame.")
            return

        machine_id = group_df['machine_id'].iloc[0] if not group_df.empty else 'unknown_machine_id'
        filename = os.path.join(output_dir, f"machine_usage_{machine_id}.csv")

        logger.debug(f"Pandas Chunk: Saving group for machine_id: {machine_id} to file: {filename}")
        logger.debug(f"Pandas Chunk: Group DataFrame head:\n{group_df.head()}")
        logger.debug(f"Pandas Chunk: Group DataFrame shape: {group_df.shape}")

        group_df.sort_values(by=["time_stamp"]).to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))
        # logger.info(f"Pandas Chunk: Appended machine usage info for {machine_id} to {filename}. Shape: {group_df.shape}")

    names = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    chunk_size = 10_000_000  # Adjust chunk size as needed based on your memory

    os.makedirs(output_directory, exist_ok=True)
    num_lines = 246_934_820
    max_iter = num_lines // chunk_size + 1
    for j, chunk_df in enumerate(pd.read_csv("../alibaba-trace/machine_usage.csv", names=names, chunksize=chunk_size, dtype={'cpu_util_percent': 'float64'})):
        logger.debug(f"Processing Pandas chunk of shape: {chunk_df.shape}")
        grouped_chunk = chunk_df.groupby("machine_id")

        for machine_id, group_chunk in grouped_chunk:
            save_group_to_csv_pandas(group_chunk, output_directory)
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    logger.info(f"CSVs saved to '{output_directory}'")


def lowest_individual_machine_utilization(options: argparse.Namespace):
    start = time.time()
    output_directory = "cluster_analysis"
    output = os.path.join(output_directory, "machine_usage_lowest_individual.csv")
    if not options.overwrite and os.path.exists(output):
        logger.warning(f"file at {output} already exists")
        return
    if os.path.exists(output):
        os.remove(output)
    dataframes = []
    missing = {192, 374, 1040, 1074, 1483, 1495, 1807, 2715, 3344, 3404, 3520}
    for machine_id in range(1, 4023 + 1):
        if machine_id in missing:
            # todo: investigate why machines are missing
            continue
        filename = os.path.join(output_directory, f"machine_usage_m_{machine_id}.csv")
        df = pd.read_csv(filename)
        # min mem footprint
        df = df[['time_stamp', 'cpu_util_percent', 'mem_util_percent']]
        dataframes.append(df)
        del df
    end = time.time()
    logger.info(f"prepared {4023 - len(missing)} machine usage files for processing in {end - start} seconds")
    df = combine_and_min_max_cpu_all_timestamps(dataframes)
    df.to_csv(output, index=False)


def test_individual_machine(options: argparse.Namespace):
    output_directory = "cluster_analysis"
    missing = {192, 374, 1040, 1074, 1483, 1495, 1807, 2715, 3344, 3404, 3520}
    dataframes = []
    for machine_id in range(1, 100 + 1):
        if machine_id in missing:
            # todo: investigate why machines are missing
            continue
        filename = os.path.join(output_directory, f"machine_usage_m_{machine_id}.csv")
        # print(filename + "\n" + str(pd.read_csv(filename)["cpu_util_percent"].value_counts(bins=10)))
        df = pd.read_csv(filename)
        print(filename)
        print(df["cpu_util_percent"].mean())
        print(df["cpu_util_percent"].median())
        # df = df[['time_stamp', 'cpu_util_percent', 'mem_util_percent']]
        # dataframes.append(df)
    # df = combine_and_min_cpu_all_timestamps(dataframes)
    # print(df["cpu_util_percent"].min())
    # print(df["cpu_util_percent"].max())


def combine_and_min_max_cpu_all_timestamps(dataframes):
    # AI generated to save time as this task sounded tedious
    """
    Combines DataFrames, including all unique timestamps, merging with merge_asof,
    and keeping the minimum cpu_util_percent for each timestamp.

    Args:
        dataframes: A list of pandas DataFrames with 'time_stamp', 'cpu_util_percent',
                    and 'mem_util_percent' columns.

    Returns:
        A single pandas DataFrame with all unique timestamps, combined data,
        minimum cpu_util_percent, and merged mem_util_percent.
    """
    start = time.time()
    if not dataframes:
        return pd.DataFrame()

    # Create a single DataFrame with all unique timestamps
    all_timestamps = pd.concat([df['time_stamp'] for df in dataframes]).unique()
    combined_df = pd.DataFrame({'time_stamp': all_timestamps}).sort_values('time_stamp').reset_index(drop=True)

    for i, df in enumerate(dataframes):
        df_sorted = df.sort_values('time_stamp')
        combined_df = pd.merge_asof(combined_df,
                                    df_sorted,
                                    on='time_stamp',
                                    suffixes=('', f'_df{i}'),
                                    direction='backward')
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, i + 1, 4023 - 11))
    # Find the minimum cpu_util_percent across all merged columns
    cpu_cols = [col for col in combined_df.columns if 'cpu_util_percent' in col]
    mem_cols = [col for col in combined_df.columns if 'mem_util_percent' in col]
    combined_df['cpu_util_percent_min'] = combined_df[cpu_cols].min(axis=1)
    combined_df['mem_util_percent_min'] = combined_df[mem_cols].min(axis=1)
    combined_df['cpu_util_percent_max'] = combined_df[cpu_cols].max(axis=1)
    combined_df['mem_util_percent_max'] = combined_df[mem_cols].max(axis=1)

    # Drop the intermediate cpu_util_percent columns
    combined_df = combined_df.drop(columns=cpu_cols)
    combined_df = combined_df.drop(columns=mem_cols)

    return combined_df


def graph_cluster_utilization(options: argparse.Namespace):
    utilization_import = import_module("0-utilization")
    create_plot = getattr(utilization_import, "create_plot")
    df = pd.read_csv('cluster_analysis/machine_usage_all.csv')
    samples = 1_000_000
    if len(df.index) > samples:
        df = df.sample(samples, replace=False)
    df.sort_values('time_stamp', inplace=True)
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    create_plot(df['time_stamp'], df['cpu_util_percent'], df['mem_util_percent'], 'machine_usage_all.csv', 'cluster_analysis',
                'cluster_analysis_graphs/cluster_utilization.png', False,
                cores_label="CPU Utilization", mem_label="Memory Utilization",
                width=10, height=6, fontsize=16)


def graph_cluster_subscription(options: argparse.Namespace):
    utilization_import = import_module("0-utilization")
    create_plot = getattr(utilization_import, "create_plot")
    df = pd.read_csv('cluster_analysis/machine_usage_subscribed.csv')
    total_cores = 4023 * 96
    total_mem = 4023
    samples = 1_000_000
    if len(df.index) > samples:
        df = df.sample(samples, replace=False)
    df.sort_values('time_stamp', inplace=True)
    matplotlib.rcParams['agg.path.chunksize'] = 10000
    create_plot(df['time_stamp'], df['cpu_subscribed'], df['mem_subscribed'], 'machine_usage_all.csv', 'cluster_analysis',
                'cluster_analysis/cluster_subscribed.png', False)


def extract_availability_information(csv_path, output, options: argparse.Namespace, cluster=False):
    start = time.time()
    chunksize = 100_000_000
    usage_df = pd.read_csv(csv_path, dtype={'time_stamp': 'int64', 'cpu_util_percent': 'float64', 'mem_util_percent': 'float64'})
    usage_df = usage_df[['time_stamp', 'cpu_util_percent', 'mem_util_percent']]
    usage_df['time_stamp'] = pd.to_numeric(usage_df['time_stamp'], errors='coerce')
    # usage_df.set_index('time_stamp')
    usage_df.sort_values('time_stamp', inplace=True)  # necessary for merge_asof even though it is already sorted
    if os.path.exists(output):
        os.remove(output)
    num_lines = 14295732
    max_iter = num_lines // chunksize + 1
    early_break = False

    for j, data in enumerate(pd.read_csv('alibaba-2018-sorted.csv', chunksize=chunksize)):
        if early_break:
            break
        # this should already be sorted

        data = data.dropna(subset=['plan_cpu', 'plan_mem'])
        chunk_for_merge: pd.DataFrame = data[['start_time', 'plan_cpu', 'plan_mem', 'instance_num']].copy()
        chunk_for_merge.rename(columns={'start_time': 'time_stamp'}, inplace=True)
        chunk_for_merge['time_stamp'] = pd.to_numeric(chunk_for_merge['time_stamp'], errors='coerce').astype('int64')

        chunk_for_merge.sort_values('time_stamp', inplace=True)
        if options.min_timestamp is not None:
            chunk_for_merge = chunk_for_merge[chunk_for_merge['time_stamp'] >= options.min_timestamp]
        if options.max_timestamp is not None:
            chunk_for_merge = chunk_for_merge[chunk_for_merge['time_stamp'] <= options.max_timestamp]
            if (chunk_for_merge['time_stamp'] > options.max_timestamp).any():
                early_break = True
        merged_df = pd.merge_asof(chunk_for_merge,
                                  usage_df,
                                  on='time_stamp',
                                  direction='backward').dropna()

        raw_available_cpu = (1 - merged_df['cpu_util_percent'] / 100) * 96
        raw_available_mem = (1 - merged_df['mem_util_percent'] / 100)
        if cluster:
            raw_available_mem *= 4023
            raw_available_mem *= 4023
        # logger.info(raw_available_cpu, merged_df['plan_cpu'].div(100))
        # import pdb
        # pdb.set_trace()
        difference_cpu = raw_available_cpu - merged_df['plan_cpu'].div(100) * merged_df['instance_num']
        difference_mem = raw_available_mem - merged_df['plan_mem'].div(100) * merged_df['instance_num']

        chunk_availability = pd.DataFrame({
            'time_stamp': merged_df['time_stamp'],
            'difference_cpu': difference_cpu,
            'difference_mem': difference_mem,
            'plan_cpu': merged_df['plan_cpu'],
            'cpu_util_percent': merged_df['cpu_util_percent'],
            'plan_mem': merged_df['plan_mem'],
            'mem_util_percent': merged_df['mem_util_percent'],
            'instance_num': merged_df['instance_num'],
            'enough_cpu': difference_cpu > 0,
            'enough_mem': difference_mem > 0,
            'enough_resources': (difference_cpu > 0) & (difference_mem > 0)
        })

        df = pd.DataFrame(chunk_availability)
        # import pdb
        # pdb.set_trace()
        df.to_csv(output, index=False, mode='a', header=not os.path.exists(output))
        logger.info(f"outputted to {output}")
        # logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))


def graph_machine_availability(options):
    output_dir = "cluster_analysis_graphs"
    os.makedirs(output_dir, exist_ok=True)
    from plot_lib import plot_scatter_min_max
    def graph_one(i):
        cluster_dir = "cluster_analysis"
        machine_availability_file = os.path.join(cluster_dir, f"cluster_availability_m_{i}.csv")
        df = pd.read_csv(machine_availability_file)
        cpu_availability_factor = df['difference_cpu'] / df['plan_cpu']
        mem_availability_factor = df['difference_mem'] / df['plan_mem']
        output_file_cpu = os.path.join(output_dir, f"cluster_availability_m_{i}_cpu.png")
        output_file_mem = os.path.join(output_dir, f"cluster_availability_m_{i}_mem.png")

        plot_scatter_min_max(df['time_stamp'], cpu_availability_factor, title="Availability factor (cpu)", output=output_file_cpu, xlabel="Timestamp", ylabel="Factor")
        plot_scatter_min_max(df['time_stamp'], mem_availability_factor, title="Availability factor (mem)", output=output_file_mem, xlabel="Timestamp", ylabel="Factor")
        logger.info(f"Extracted to {output_file_cpu}, {output_file_mem}")

    if options.n is not None:
        graph_one(options.n)
    else:
        for i in range(0, 4023 + 1):
            graph_one(i)
            break


def find_closest_value_left_sorted_vectorized(df, column_name, target_value):
    """
    To save time, I AI generated this function as it is a tedious one to write. From a glance, this looks correct

    Finds the closest value to the left in a *sorted* DataFrame column using vectorized operations.

    Assumes DataFrame is already sorted by column_name.

    Args:
        df (pd.DataFrame): The input DataFrame (must be sorted by column_name).
        column_name (str): The name of the column to search in.
        target_value: The value to find the closest match for.

    Returns:
        pd.Series or None: The row (as a Pandas Series) with the closest value,
                          or None if no suitable row is found.
    """

    # Find the index where 'target_value' *would be* inserted to maintain sorted order (binary search)
    insertion_point = df[column_name].searchsorted(target_value, side='left')

    if insertion_point == 0:
        # Target is smaller than all values, or closest is the first value (if non-empty)
        if df.empty:
            return None
        else:
            return df.iloc[0]  # Return the first row as closest from the left (smallest value)
    elif insertion_point == len(df):
        # Target is larger than all values, closest is the last value
        return df.iloc[-1]
    else:
        # Check values at insertion point and the one before it
        possible_closest_indices = [insertion_point - 1]
        if insertion_point < len(df):
            possible_closest_indices.append(insertion_point)

        possible_closest_rows = df.iloc[possible_closest_indices]

        # Vectorized calculation of differences for all possible closest rows
        differences = abs(possible_closest_rows[column_name] - target_value)

        # Find the index of the row with the minimum difference using idxmin()
        closest_row_index = differences.idxmin()

        # Get the closest row using .loc (label-based indexing)
        closest_row = possible_closest_rows.loc[closest_row_index]

        return closest_row


# @limit_dask_client_decorator(10, 30)
def examine_cluster_usage_during_scheduling(options: argparse.Namespace):
    """
    For any given job, look at when it was scheduled and the associated utilization
    of the cluster. Map the two values and put into a CSV value to graph.
    """
    # first get real usage of machines
    output_directory = "cluster_analysis"
    os.makedirs(output_directory, exist_ok=True)
    cluster_utilization = os.path.join(output_directory, "machine_usage_all.csv")
    if options.overwrite is True or not os.path.exists(cluster_utilization):
        start = time.time()
        logger.warning(f"only one file under {output_directory} is checked for existence, ensure others do not exist")

        # todo: the below didn't work, it didn't go through the entire CSV for some reason
        # it would be nice to get this working as dask is faster

        # names = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
        # machine_usage_df = dd.read_csv("../alibaba-trace/machine_usage.csv", names=names, dtype={'cpu_util_percent': 'float64'})
        #
        # def save_group_to_csv_partition(group_df, output_dir):
        #     machine_id = group_df['machine_id'].iloc[0] if not group_df.empty else 'unknown_machine_id'  # Get machine_id from the group
        #     filename = os.path.join(output_dir, f"machine_usage_{machine_id}.csv")
        #     # group_df.sort_values(by="time_stamp").to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))
        #     group_df.to_csv(filename, index=False, mode='a', header=not os.path.exists(filename))
        #     return machine_id
        #
        # grouped_df = machine_usage_df.groupby("machine_id")
        # grouped_df.apply(
        #     save_group_to_csv_partition,
        #     output_dir=output_directory,
        #     # meta=machine_usage_df._meta
        # ).compute()
        if options.dont_skip_machines:
            individual_machine_utilization(output_directory, options)

        # the above will throw per machine utilization into each individual CSV file.
        old_utilization_port(cluster_utilization, options)
        # the above will compute cluster util into a CSV file

        requested_utilization = os.path.join(output_directory, "machine_usage_subscribed.csv")
        requested_utilization_port(requested_utilization, options)
        # the above will calculate the subscribed utilization
        end = time.time()
        logger.info(f"Completed extraction of utilization data in {end - start} seconds")

    csv_files = glob.glob(os.path.join(output_directory, '*.csv'))
    start = time.time()
    for csv_path in csv_files:
        if not os.path.basename(csv_path).startswith("machine_usage"):
            continue
        id = os.path.basename(csv_path).split("machine_usage_")[1].split(".")[0]
        if id == "a":
            # bad value??
            continue
        if id == "all":
            # this is on the entire cluster
            output = os.path.join(output_directory, "cluster_availability.csv")
            extract_availability_information(csv_path, output, options, cluster=True)
            graph_cluster_availability(options)
        elif id == "subscribed":
            # this is the subscribed number of resources
            pass
        else:
            continue
            # this is an individual machine
            # output = os.path.join(output_directory, f"cluster_availability_{id}.csv")
            # extract_availability_information(csv_path, output, options)
    end = time.time()


def graph_cluster_availability(options: argparse.Namespace):
    output_directory = "cluster_analysis"
    cluster_availability = os.path.join(output_directory, "cluster_availability.csv")

    utilization_import = import_module("0-utilization")
    create_plot = getattr(utilization_import, "create_plot")

    df = pd.read_csv(cluster_availability)
    samples = 1_000_000
    if len(df.index) > samples:
        df = df.sample(samples, replace=False)
    df.sort_values('time_stamp', inplace=True)

    total = len(df.index)
    total_enough_cpu = df[["enough_cpu"]].sum()
    total_enough_mem = df[["enough_mem"]].sum()
    logger.info(f"enough cpu: {total_enough_cpu}, enough mem: {total_enough_mem}, total: {total}")


def extract_availability_information_of_machines(options: argparse.Namespace):
    output_directory = "cluster_analysis"
    csv_files = glob.glob(os.path.join(output_directory, '*.csv'))
    start = time.time()
    extracted_out = []
    total = 4023
    i = 0
    for csv_path in csv_files:
        if not os.path.basename(csv_path).startswith("machine_usage"):
            continue
        id = os.path.basename(csv_path).split("machine_usage_")[1].split(".")[0]
        if id == "a":
            # bad value??
            continue
        if id == "all":
            continue
        elif id == "subscribed":
            continue
        else:
            # this is an individual machine
            output = os.path.join(output_directory, f"cluster_availability_{id}.csv")
            if os.path.exists(output):
                os.remove(output)
            extract_availability_information(csv_path, output, options, cluster=False)
            extracted_out.append(output)
            i += 1
            logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, i + 1, total))
            break

    end = time.time()
    logger.info(f"Took {end - start} seconds to extract to {','.join(extracted_out)}")


def examine_job_qdelay_during_scheduling(options: argparse.Namespace):
    """
    For any given job, look at its queueing delay and if there exists a machine available
    during scheduling. Map the delay and availability and put into CSV.
    """
    # load csv with batch task (scheduled info) and queueing delay (qdelay per instance)
    # and merge on task name and job name
    # then iterate through all instances and calculate how long after initially scheduled
    # before space is available (per machine and per cluster)
    # todo: this is per instance but per task could be helpful too
    # i may want to merge csvs over timestamps
    qdelay_file = "batch_instance_analysis/queueing_delays_dag.csv"

    utilization_file = "cluster_analysis/machine_usage_m_1275.csv"

    utilization_df = pd.read_csv(utilization_file)
    utilization_df.sort_values("time_stamp", inplace=True)

    chunksize = 1_000_000
    # columns = ["timestamp", "queueing_delay", "completion_time", "job_name", "task_name", "job_length"]
    for j, data in enumerate(pd.read_csv(qdelay_file, chunksize=chunksize)):
        data.sort_values('time_stamp', inplace=True)

        merged_df = pd.merge_asof(data, utilization_df, on='time_stamp', direction='backward')
        # alternative, merge on timestamp, for instance with queueing delay, find how long until an available machine/cluster space is available
        for row in merged_df.to_numpy():
            pass
    pass


def graph_machine_utilization_all(options):
    create_plot = getattr(utilization_import, "create_plot_one_per")

    cluster_dir = "cluster_analysis"
    output_dir = "cluster_analysis_graphs"
    machine_usage_all_csv = os.path.join(cluster_dir, "machine_usage_all.csv")
    df = pd.read_csv(machine_usage_all_csv)
    output = os.path.join(output_dir, "machine_usage_all.png")
    os.makedirs(output_dir, exist_ok=True)

    create_plot(df['time_stamp'], df['cpu_util_percent'], df['mem_util_percent'], input_file_name=os.path.basename(machine_usage_all_csv), output_directory=output_dir,
                write_config=False,
                cores_label="CPU usage (total)", mem_label="Mem usage (total)", output=output, width=20, height=6)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Run a specified operation.")
    parser.add_argument("--operation", "-o", type=str, help="Name of the operation (function) to execute.")
    parser.add_argument("--overwrite", action="store_true", default=-False)
    parser.add_argument("--min-timestamp", type=int, default=86400 * 2)  # todo: temp
    parser.add_argument("--max-timestamp", type=int, default=86400 * 3)  # todo: temp
    parser.add_argument("--dont-skip-machines", type=bool, default=False)
    parser.add_argument("-n", type=int, default=None)
    options = parser.parse_args(args)

    operation_name = options.operation

    if options.max_timestamp == 0:
        options.max_timestamp = None
    if options.min_timestamp == 0:
        options.min_timestamp = None

    if operation_name:
        try:
            operation_function = globals()[operation_name]
            operation_function(options)
        except KeyError:
            print(f"Function {operation_name} not found.")
            raise
        except Exception:
            print(f"Could not run function {operation_name}")
            raise
    else:
        print("Specify an function using --operation.")


# https://stackoverflow.com/questions/55057957/an-attempt-has-been-made-to-start-a-new-process-before-the-current-process-has-f
if __name__ == "__main__":
    main()

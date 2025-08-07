import argparse
import glob
import logging
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from batch_instance_analyze import calcProcessTime

from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

output_directory = "timeline"
os.makedirs(output_directory, exist_ok=True)


def extract_batch_instance_all(options: argparse.Namespace):
    batch_file = "../alibaba-trace/batch_instance_sort.csv"
    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}
    names = list(names_to_types.keys())
    start = time.time()
    # code copied from extract_batch_instance_type
    chunksize = 100_000_000
    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1
    start_timestamp, end_timestamp = options.timestamp_range.split('-')
    output_path = os.path.join(output_directory, options.output or "batch_instance.csv")
    if os.path.exists(output_path):
        os.remove(output_path)
    for j, data in enumerate(pd.read_csv(batch_file, names=names, engine="c", chunksize=chunksize, header=None)):
        out_df = data[(data[options.timestamp_name] >= int(start_timestamp)) & (data[options.timestamp_name] < int(end_timestamp))]
        out_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    end = time.time()
    logger.info("done in %s seconds" % (end - start))

def machine_id_metrics(options: argparse.Namespace):
    batch_file = "../alibaba-trace/batch_instance_sort.csv"
    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}
    names = list(names_to_types.keys())
    start = time.time()
    # code copied from extract_batch_instance_type
    chunksize = 100_000_000
    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1
    max_iterations = 3
    a = None
    for j, data in enumerate(pd.read_csv(batch_file, names=names, engine="c", chunksize=chunksize, header=None)):
        # if j > max_iterations:
        #     break
        S = data.value_counts("machine_id")
        if a is None:
            a = S
        else:
            a = a.add(S, fill_value=0)
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    print(a)
    a.to_csv("timeline/metrics.csv")
    end = time.time()
    logger.info("done in %s seconds" % (end - start))

def extract_batch_instance(options: argparse.Namespace):
    batch_file = "../alibaba-trace/batch_instance_sort.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    start = time.time()

    # code copied from extract_batch_instance_type
    chunksize = 100_000_000

    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1

    target_timestamp = options.target

    output_path = os.path.join(output_directory, options.output or "batch_instance.csv")
    os.remove(output_path)
    rows_above_below = options.range or 10
    for j, data in enumerate(pd.read_csv(batch_file, names=names, engine="c", chunksize=chunksize, header=None)):
        mask = data[options.timestamp_name] > target_timestamp
        if mask.any():
            # one of the start time values is greater than this, get the surrounding
            first_idx = np.argmax(mask) % chunksize
            start_idx = first_idx - rows_above_below
            if start_idx < 0:
                logger.info(f"start idx hit first row, could not go further")
                start_idx = 0
            end_idx = first_idx + rows_above_below + 1
            if end_idx >= len(data):
                logger.info(f"end idx hit last row, could not go further")
                end_idx = len(data) - 1
            out_df = data.iloc[start_idx:end_idx]
            out_df.to_csv(output_path, header=not os.path.exists(output_path), index=False)
            break
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    end = time.time()
    logger.info("done in %s seconds" % (end - start))


def extract_request(options: argparse.Namespace):
    batch_file = "alibaba-2018-sorted.csv"

    target_timestamp = options.target
    df = pd.read_csv(batch_file)
    df.sort_values(by="start_time", inplace=True)
    mask = df["start_time"] > target_timestamp
    start = time.time()
    rows_above_below = options.range or 10
    if mask.any():
        # one of the start time values is greater than this, get the surrounding
        first_idx = np.argmax(mask)
        start_idx = first_idx - rows_above_below
        if start_idx < 0:
            logger.info(f"start idx hit first row, could not go further")
            start_idx = 0
        end_idx = first_idx + rows_above_below + 1
        if end_idx >= len(df):
            logger.info(f"end idx hit last row, could not go further")
            end_idx = len(df) - 1
        out_df = df.iloc[start_idx:end_idx]
        out_df.to_csv(os.path.join(output_directory, "request.csv"), index=False)
    else:
        logger.info(f"no timestamp found for {target_timestamp}")
    end = time.time()
    logger.info("done in %s seconds" % (end - start))


def extract_request_name(options: argparse.Namespace):
    batch_file = "alibaba-2018-sorted.csv"

    target_timestamp = options.target
    df = pd.read_csv(batch_file)
    df.sort_values(by="start_time", inplace=True)
    assert options.job_name is not None
    assert options.task_name is not None
    mask = (df["job_name"] == options.job_name) & (df["task_name"] == options.task_name)
    start = time.time()
    rows_above_below = options.range or 10
    if mask.any():
        # one of the start time values is greater than this, get the surrounding
        first_idx = np.argmax(mask)
        start_idx = first_idx - rows_above_below
        if start_idx < 0:
            logger.info(f"start idx hit first row, could not go further")
            start_idx = 0
        end_idx = first_idx + rows_above_below + 1
        if end_idx >= len(df):
            logger.info(f"end idx hit last row, could not go further")
            end_idx = len(df) - 1
        out_df = df.iloc[start_idx:end_idx]
        out_df.to_csv(os.path.join(output_directory, "request_name.csv"), index=False)
    else:
        logger.info(f"no timestamp found for {target_timestamp}")
    end = time.time()
    logger.info("done in %s seconds" % (end - start))


def extract_batch_name(options: argparse.Namespace):
    batch_file = "../alibaba-trace/batch_instance_sort.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    assert options.job_name is not None
    assert options.task_name is not None
    start = time.time()
    results = []

    chunksize = 100_000_000

    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1

    for j, data in enumerate(pd.read_csv(batch_file, names=names, engine="c", chunksize=chunksize, header=None)):
        mask = (data["job_name"] == options.job_name) & (data["task_name"] == options.task_name)
        if mask.any():
            # one of the start time values is greater than this, get the surrounding
            results.append(data[mask])
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    df = pd.concat(results)
    df.to_csv(os.path.join(output_directory, "batch_job_task_name.csv"), index=False)
    end = time.time()
    logger.info("done in %s seconds" % (end - start))


utilization_import = import_module("0-utilization")
create_plot = getattr(utilization_import, "create_plot")
from scipy.ndimage import gaussian_filter1d


def graph_machine_usage_individual(csv_path, outdir, start_timestamp, end_timestamp, sigma):
    """Processes a single CSV file."""
    if not os.path.basename(csv_path).startswith("machine_usage"):
        return

    id = os.path.basename(csv_path).split("machine_usage_")[1].split(".")[0]
    if id in ("a", "all", "subscribed", "lowest_individual"):
        return

    machine_utilization_file = os.path.join(os.path.dirname(csv_path), f"machine_usage_{id}.csv")
    df = pd.read_csv(machine_utilization_file)
    df.sort_values(by=['time_stamp'], inplace=True)
    df.drop_duplicates(subset=["time_stamp"], inplace=True)
    if start_timestamp is not None and end_timestamp is not None:
        mask = (df["time_stamp"] > start_timestamp) & (df["time_stamp"] < end_timestamp)
        df = df[mask]

    output = os.path.join(outdir, f"machine_usage_{id}.png")

    try:
        cpu_smoothed = gaussian_filter1d(df['cpu_util_percent'], sigma)
        mem_smoothed = gaussian_filter1d(df['mem_util_percent'], sigma)
    except KeyError as e:
        print(f"path at {csv_path} did not have a valid key for the following error: {e}")
        return

    create_plot(df['time_stamp'], cpu_smoothed, mem_smoothed,
                input_file_name=os.path.basename(machine_utilization_file),
                output_directory=outdir,
                write_config=False,
                cores_label="CPU usage (total)", mem_label="Mem usage (total)", output=output, width=10, height=6,
                num_xticks=20)


def extract_machine_utilization_graphs(options: argparse.Namespace):
    output_directory = "timeline_graphs"
    machine_utilization_directory = "cluster_analysis"
    os.makedirs(output_directory, exist_ok=True)

    sigma = 3

    csv_files = glob.glob(os.path.join(machine_utilization_directory, '*.csv'))
    start = time.time()
    max_iter = len(csv_files)
    start_time = options.start
    end_time = options.end
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    output_directory = "timeline_graphs"
    os.makedirs(output_directory, exist_ok=True)
    output = os.path.join(output_directory, f"machine_usage_coalesced.png")

    batch_instances = pd.read_csv('timeline/batch_job_task_name_j_1915435.csv')
    machine_ids = list(batch_instances["machine_id"])
    for csv_path in csv_files:
        if not os.path.basename(csv_path).startswith("machine_usage"):
            continue
        id = os.path.basename(csv_path).split("machine_usage_")[1].split(".")[0]
        if id in ("a", "all", "subscribed", "lowest_individual"):
            continue

        if id not in machine_ids:
            continue

        machine_utilization_file = os.path.join(os.path.dirname(csv_path), f"machine_usage_{id}.csv")
        df = pd.read_csv(machine_utilization_file)
        df.sort_values(by=['time_stamp'], inplace=True)
        df.drop_duplicates(subset=["time_stamp"], inplace=True)
        if options.start is not None and options.end is not None:
            mask = (df["time_stamp"] > options.start) & (df["time_stamp"] < options.end)
            df = df[mask]
        cpu_smoothed = gaussian_filter1d(df['cpu_util_percent'], sigma)
        mem_smoothed = gaussian_filter1d(df['mem_util_percent'], sigma)
        color = 'tab:red'
        ax1.set_xlabel('Timestamp (seconds)')
        ax1.set_ylabel('CPU usage (total)', color=color)
        ax1.plot(df['time_stamp'], cpu_smoothed, color=color, alpha=0.5, linewidth=0.2)
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:blue'
        ax2.set_ylabel('Mem usage (total)', color=color)  # we already handled the x-label with ax1
        ax2.plot(df['time_stamp'], mem_smoothed, color=color, alpha=0.5, linewidth=0.2)
        ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title('Utilization Graph')
    num_xticks = True
    if num_xticks:
        ax1.locator_params(axis='x', nbins=20)
        ax1.tick_params(axis='x', labelrotation=45, labelright=True)
    if num_xticks:
        ax2.locator_params(axis='x', nbins=20)
        ax1.tick_params(axis='x', labelrotation=45, labelright=True)
    plt.savefig(output)

    plt.close()
    end = time.time()

    print(f"took {end - start} seconds")


def extract_machine_utilization_graphs_average(options: argparse.Namespace):
    output_directory = "timeline_graphs"
    machine_utilization_directory = "cluster_analysis"
    os.makedirs(output_directory, exist_ok=True)
    csv_files = glob.glob(os.path.join(machine_utilization_directory, '*.csv'))
    start = time.time()
    max_iter = len(csv_files)
    start_time = options.start
    end_time = options.end
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    output_directory = "timeline_graphs"
    os.makedirs(output_directory, exist_ok=True)
    output = os.path.join(output_directory, f"machine_usage_coalesced.png")

    batch_instances = pd.read_csv(f'timeline/batch_job_task_name_{options.job_name}.csv')
    machine_ids = set(batch_instances["machine_id"])

    data_sets = []
    for csv_path in csv_files:
        if not os.path.basename(csv_path).startswith("machine_usage"):
            continue
        id = os.path.basename(csv_path).split("machine_usage_")[1].split(".")[0]
        if id in ("a", "all", "subscribed", "lowest_individual"):
            continue

        if id not in machine_ids:
            continue

        machine_utilization_file = os.path.join(os.path.dirname(csv_path), f"machine_usage_{id}.csv")
        df = pd.read_csv(machine_utilization_file)
        df.sort_values(by=['time_stamp'], inplace=True)
        df.drop_duplicates(subset=["time_stamp"], inplace=True)
        if options.start is not None and options.end is not None:
            mask = (df["time_stamp"] > options.start) & (df["time_stamp"] < options.end)
            df = df[mask]
            df = df[['time_stamp', 'cpu_util_percent', 'mem_util_percent']]
        data_sets.append(df)

    min_x = float('inf')
    max_x = 0
    for df in data_sets:
        x = df["time_stamp"]
        if len(x) > 0:
            min_x = min(min_x, min(x))
            max_x = max(max_x, max(x))
    common_x = np.linspace(min_x, max_x, num=1000)
    interp_mem = []
    interp_cpu = []
    for df in data_sets:
        x = df["time_stamp"]
        if len(x) == 0:
            # empty, maybe missing data
            continue
        cpu = df["cpu_util_percent"]
        mem = df["mem_util_percent"]
        f_cpu = interp1d(x, cpu, kind='linear', fill_value="extrapolate")
        f_mem = interp1d(x, mem, kind='linear', fill_value="extrapolate")
        interp_mem.append(f_mem(common_x))
        interp_cpu.append(f_cpu(common_x))

    top10 = False
    top_percent = 1/100
    if top10:
        top_10_percent_means_cpu = []
        top_10_percent_means_mem = []
        interp_cpu = np.array(interp_cpu)
        interp_mem = np.array(interp_mem)
        for x_index in range(len(common_x)):
            cpu_y_values_at_x = interp_cpu[:, x_index]
            mem_y_values_at_x = interp_mem[:, x_index]
            cpu_y_values_at_x = cpu_y_values_at_x[cpu_y_values_at_x <= 100]
            mem_y_values_at_x = mem_y_values_at_x[mem_y_values_at_x >= 0]
            sorted_cpu_y_values = np.sort(cpu_y_values_at_x)[::-1]  # Sort descending
            sorted_mem_y_values = np.sort(mem_y_values_at_x)[::-1]  # Sort descending
            top_10_percent_count_cpu = int(len(sorted_cpu_y_values) * top_percent)
            top_10_percent_count_mem = int(len(sorted_mem_y_values) * top_percent)
            top_10_percent_cpu_y_values = sorted_cpu_y_values[:top_10_percent_count_cpu]
            top_10_percent_mem_y_values = sorted_mem_y_values[:top_10_percent_count_mem]
            top_10_percent_mean_cpu = np.mean(top_10_percent_cpu_y_values)
            top_10_percent_mean_mem = np.mean(top_10_percent_mem_y_values)
            top_10_percent_means_cpu.append(top_10_percent_mean_cpu)
            top_10_percent_means_mem.append(top_10_percent_mean_mem)
        cpu = top_10_percent_means_cpu
        mem = top_10_percent_means_mem
    else:
        df_cpu = pd.DataFrame(np.array(interp_cpu).T, index=common_x)
        df_mem = pd.DataFrame(np.array(interp_mem).T, index=common_x)
        cpu = df_cpu.mean(axis=1)
        mem = df_mem.mean(axis=1)

    sigma = 3
    mem = gaussian_filter1d(mem, sigma)
    cpu = gaussian_filter1d(cpu, sigma)

    color = 'tab:red'
    ax1.set_xlabel('Timestamp (seconds)')
    ax1.set_ylabel('CPU usage (total)', color=color)
    ax1.plot(common_x, cpu, color=color, alpha=1, linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color)
    color = 'tab:blue'
    ax2.set_ylabel('Mem usage (total)', color=color)  # we already handled the x-label with ax1
    ax2.plot(common_x, mem, color=color, alpha=1, linewidth=1)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title('Utilization Graph')
    num_xticks = True
    if num_xticks:
        ax1.locator_params(axis='x', nbins=20)
        ax1.tick_params(axis='x', labelrotation=45, labelright=True)
    if num_xticks:
        ax2.locator_params(axis='x', nbins=20)
        ax1.tick_params(axis='x', labelrotation=45, labelright=True)
    plt.savefig(output)

    plt.close()
    end = time.time()

    print(f"took {end - start} seconds")

def extract_machine_utilization_graphs_all_in_one(options: argparse.Namespace):
    output_directory = "timeline_graphs"
    machine_utilization_directory = "cluster_analysis"
    os.makedirs(output_directory, exist_ok=True)

    sigma = 3

    csv_files = glob.glob(os.path.join(machine_utilization_directory, '*.csv'))
    start = time.time()
    max_iter = len(csv_files)
    start_time = options.start
    end_time = options.end

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(graph_machine_usage_individual, csv_path, output_directory, start_time, end_time, sigma) for csv_path in csv_files]
        for j, future in enumerate(as_completed(futures)):
            future.result()

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Run a specified operation.")
    parser.add_argument("--operation", "-o", type=str, help="Name of the operation (function) to execute.")
    parser.add_argument("--target", "-t", type=int)
    parser.add_argument("--overwrite", action="store_true", default=-False)
    parser.add_argument("--range", "-r", type=int)
    parser.add_argument("--job_name", "--job")
    parser.add_argument("--task_name", "--task")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--timestamp_name", default="start_time")
    parser.add_argument("--output")
    parser.add_argument("--timestamp_range")
    options = parser.parse_args(args)

    operation_name = options.operation

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


if __name__ == "__main__":
    main()

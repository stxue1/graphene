import argparse
import logging
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from batch_instance_analyze import calcProcessTime

plot_import = import_module("3-plot")
calculate_cdf = getattr(plot_import, "calculate_cdf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

utilization_import = import_module("0-utilization")

get_clean_filename = getattr(utilization_import, "get_clean_filename")
get_plot_filename = getattr(utilization_import, "get_plot_filename")
get_store_filename = getattr(utilization_import, "get_store_filename")

create_plot = getattr(utilization_import, "create_plot_one_per")


def graph_queueing_delay_over_completion(options: argparse.Namespace):
    graph_queueing_delay(options, "completion_time", "queueing_delay", "job_length", "Completion Time in seconds (log)", "Queueing Delay in seconds (log)",
                         "queueing_delay_over_completion.png", log=True)


def graph_queueing_delay_over_cpu_max(options: argparse.Namespace):
    graph_queueing_delay(options, "cpu_max", "queueing_delay", "job_length", "cpu max", "queueing delay",
                         "queueing_delay_over_cpu_max.png")


def graph_queueing_delay_over_cpu_avg(options: argparse.Namespace):
    graph_queueing_delay(options, "cpu_avg", "queueing_delay", "job_length", "cpu avg", "queueing delay",
                         "queueing_delay_over_cpu_avg.png")


def graph_queueing_delay_over_mem_max(options: argparse.Namespace):
    graph_queueing_delay(options, "mem_max", "queueing_delay", "job_length", "mem max", "queueing delay",
                         "queueing_delay_over_mem_max.png")


def graph_queueing_delay_over_mem_avg(options: argparse.Namespace):
    graph_queueing_delay(options, "mem_avg", "queueing_delay", "job_length", "mem avg", "queueing delay",
                         "queueing_delay_over_mem_avg.png")


def graph_queueing_delay_over_lowest_machine_cpu(options: argparse.Namespace):
    graph_queueing_delay(options, "at_req_lowest_machine_cpu_util_percent", "queueing_delay", "job_length", "lowest machine cpu", "queueing delay",
                         "queueing_delay_over_lowest_machine_cpu.png")


def graph_queueing_delay_over_lowest_machine_mem(options: argparse.Namespace):
    graph_queueing_delay(options, "at_req_lowest_machine_mem_util_percent", "queueing_delay", "job_length", "lowest machine mem", "queueing delay",
                         "queueing_delay_over_lowest_machine_mem.png")


def graph_queueing_delay_over_highest_machine_cpu(options: argparse.Namespace):
    graph_queueing_delay(options, "at_req_highest_machine_cpu_util_percent", "queueing_delay", "job_length", "highest machine cpu", "queueing delay",
                         "queueing_delay_over_highest_machine_cpu.png")


def graph_queueing_delay_over_highest_machine_mem(options: argparse.Namespace):
    graph_queueing_delay(options, "at_req_highest_machine_mem_util_percent", "queueing_delay", "job_length", "highest machine mem", "queueing delay",
                         "queueing_delay_over_highest_machine_mem.png")


def graph_queueing_delay_over_cluster_cpu(options: argparse.Namespace):
    graph_queueing_delay(options, "at_req_cluster_cpu_util_percent", "queueing_delay", "job_length", "cluster cpu", "queueing delay",
                         "queueing_delay_over_cluster_cpu.png")


def graph_queueing_delay_over_cluster_mem(options: argparse.Namespace):
    graph_queueing_delay(options, "at_req_cluster_mem_util_percent", "queueing_delay", "job_length", "cluster mem", "queueing delay",
                         "queueing_delay_over_cluster_mem.png")


def graph_queueing_delay_over_plan_cpu(options: argparse.Namespace):
    graph_queueing_delay(options, "plan_cpu", "queueing_delay", None, "plan cpu", "queueing delay",
                         "queueing_delay_over_plan_cpu.png", remove_duplicates=True, log=True)


def graph_queueing_delay_over_plan_mem(options: argparse.Namespace):
    graph_queueing_delay(options, "plan_mem", "queueing_delay", None, "plan mem", "queueing delay",
                         "queueing_delay_over_plan_mem.png", remove_duplicates=True)


def graph_queueing_delay_over_instance_num(options: argparse.Namespace):
    graph_queueing_delay(options, "instance_num", "queueing_delay", None, "Number of Instances", "Queueing Delay in seconds",
                         "queueing_delay_over_instance_num.png", remove_duplicates=True)


def graph_job_length_over_instance_num(options: argparse.Namespace):
    graph_queueing_delay(options, "job_length", "instance_num", None, "job_length", "instance_num",
                         "job_length_over_instance_num.png", log=True, remove_duplicates=True)


def graph_a_bunch(options: argparse.Namespace):
    graph_queueing_delay_over_lowest_machine_cpu(options)
    graph_queueing_delay_over_lowest_machine_mem(options)
    graph_queueing_delay_over_highest_machine_cpu(options)
    graph_queueing_delay_over_highest_machine_mem(options)
    graph_queueing_delay_over_cluster_cpu(options)
    graph_queueing_delay_over_cluster_mem(options)
    graph_queueing_delay_over_plan_cpu(options)
    graph_queueing_delay_over_plan_mem(options)
    graph_queueing_delay_over_instance_num(options)


def iterate_queueing_delay(options: argparse.Namespace):
    # queueing_delay = "batch_instance_analysis/queueing_delays_merged.csv"
    num_points = options.num_points or 1_000_000
    dag = options.dag or True
    if dag:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_dag_{num_points}.csv")
        queueing_delay = "batch_instance_analysis/queueing_delays_dag.csv"
    else:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_{num_points}.csv")
        queueing_delay = "batch_instance_analysis/queueing_delays.csv"
    output_dir = "queueing_delay_graphs"
    os.makedirs(output_dir, exist_ok=True)
    # output = os.path.join(output_dir, output)
    num_lines = 1_228_679_841
    n_bins = 100
    chunksize = 100_000_000
    max_iter = num_lines // chunksize + 1
    size = int(num_points / max_iter)
    logger.info(f"looping {max_iter} times and selecting {size} data pts each time")
    highest = 0
    for j, data in enumerate(
            pd.read_csv(queueing_delay, engine="c", chunksize=chunksize)):
        cur_high = max(data["at_req_lowest_machine_cpu_util_percent"])
        if cur_high > highest:
            highest = cur_high
        # logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    logger.info(f"highest: {highest}")


def just_sample(options: argparse.Namespace):
    num_points = options.num_points or 1_000_000
    cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_dag_{num_points}.csv")
    queueing_delay = "batch_instance_analysis/queueing_delays_dag.csv"

    output_dir = "queueing_delay_graphs"
    os.makedirs(output_dir, exist_ok=True)
    num_lines = 1_228_679_841
    n_bins = 100

    start = time.time()
    if os.path.exists(cache_file):
        os.remove(cache_file)
    chunksize = 100_000_000
    max_iter = num_lines // chunksize + 1
    size = int(num_points / max_iter)
    logger.info(f"looping {max_iter} times and selecting {size} data pts each time")
    for j, data in enumerate(
            pd.read_csv(queueing_delay, engine="c", chunksize=chunksize)):
        sampled_data = data.sample(size, replace=False)
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
        sampled_data.to_csv(cache_file, header=not os.path.exists(cache_file), mode="a")


def remove_negative(l):
    for e in l:
        if e < 0:
            return False
    return True


def remove_negative_from_arrays(*arrays):
    if not arrays or len(arrays) == 0:
        return None

    zipped_data = zip(*arrays)
    filtered_data = filter(remove_negative, zipped_data)
    sorted_data = sorted(filtered_data)

    return tuple(list(t) for t in zip(*sorted_data))


def graph_queueing_delay(options: argparse.Namespace, xaxis, yaxis, coloraxis, xtitle, ytitle, output, log=True, remove_duplicates=False):
    # queueing_delay = "batch_instance_analysis/queueing_delays_merged.csv"
    num_points = options.num_points or 1_000_000
    dag = options.dag or True
    if dag:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_dag_{num_points}.csv")
        queueing_delay = "batch_instance_analysis/queueing_delays_dag.csv"
    else:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_{num_points}.csv")
        queueing_delay = "batch_instance_analysis/queueing_delays.csv"
    output_dir = "queueing_delay_graphs"
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, output)
    num_lines = 1_228_679_841
    n_bins = 100

    start = time.time()
    if os.path.exists(cache_file) and not options.overwrite:
        df = pd.read_csv(cache_file)
        if remove_duplicates:
            df.drop_duplicates(["job_name", "task_name"], inplace=True)
        timestamps = df["timestamp"]
        qdelay_pts = df["queueing_delay"]
        completion_times = df["completion_time"]
        job_lengths = df["job_length"]
        cpu_avgs = df["cpu_avg"]
        cpu_maxes = df["cpu_max"]
        mem_avgs = df["mem_avg"]
        mem_maxes = df["mem_max"]
        at_req_cpu = df["at_req_cpu_subscribed"]
        at_req_mem = df["at_req_mem_subscribed"]
        at_req_lowest_machine_cpu = df["at_req_lowest_machine_cpu_util_percent"]
        at_req_lowest_machine_mem = df["at_req_lowest_machine_mem_util_percent"]
        at_req_highest_machine_cpu = df["at_req_highest_machine_cpu_util_percent"]
        at_req_highest_machine_mem = df["at_req_highest_machine_mem_util_percent"]
        at_req_cluster_cpu_util_percent = df["at_req_cluster_cpu_util_percent"]
        at_req_cluster_mem_util_percent = df["at_req_cluster_mem_util_percent"]
        plan_cpu = df["plan_cpu"] * df["instance_num"]
        plan_mem = df["plan_mem"] * df["instance_num"]
        instance_num = df["instance_num"]
        del df
    else:
        if os.path.exists(cache_file):
            os.remove(cache_file)
        logger.info("run just_sample first")
        return

    # timestamps, qdelay_pts = (list(t) for t in zip(*sorted(filter(remove_negative, zip(timestamps, qdelay_pts)))))

    (timestamps, completion_times, qdelay_pts, job_lengths, cpu_avgs, cpu_maxes, mem_avgs, mem_maxes,
     at_req_cpu, at_req_mem, at_req_lowest_machine_cpu, at_req_lowest_machine_mem, at_req_highest_machine_cpu, at_req_highest_machine_mem,
     at_req_cluster_cpu_util_percent, at_req_cluster_mem_util_percent, plan_cpu, plan_mem, instance_num) = remove_negative_from_arrays(timestamps, completion_times,
                                                                                                                                       qdelay_pts, job_lengths, cpu_avgs, cpu_maxes,
                                                                                                                                       mem_avgs, mem_maxes,
                                                                                                                                       at_req_cpu, at_req_mem,
                                                                                                                                       at_req_lowest_machine_cpu,
                                                                                                                                       at_req_lowest_machine_mem,
                                                                                                                                       at_req_highest_machine_cpu,
                                                                                                                                       at_req_highest_machine_mem,
                                                                                                                                       at_req_cluster_cpu_util_percent,
                                                                                                                                       at_req_cluster_mem_util_percent,
                                                                                                                                       plan_cpu, plan_mem, instance_num
                                                                                                                                       )
    mapping = {
        "completion_time": completion_times,
        "queueing_delay": qdelay_pts,
        "job_length": job_lengths,
        "timestamp": timestamps,
        "cpu_avg": cpu_avgs,
        "cpu_max": cpu_maxes,
        "mem_avg": mem_avgs,
        "mem_max": mem_maxes,
        "at_req_cpu_subscribed": at_req_cpu,
        "at_req_mem_subscribed": at_req_mem,
        "at_req_lowest_machine_cpu_util_percent": at_req_lowest_machine_cpu,
        "at_req_lowest_machine_mem_util_percent": at_req_lowest_machine_mem,
        "at_req_highest_machine_cpu_util_percent": at_req_highest_machine_cpu,
        "at_req_highest_machine_mem_util_percent": at_req_highest_machine_mem,
        "at_req_cluster_cpu_util_percent": at_req_cluster_cpu_util_percent,
        "at_req_cluster_mem_util_percent": at_req_cluster_mem_util_percent,
        "plan_cpu": plan_cpu, "plan_mem": plan_mem, "instance_num": instance_num
    }
    xaxis_pts = mapping[xaxis]
    yaxis_pts = mapping[yaxis]
    coloraxis_pts = mapping.get(coloraxis)

    print(output)
    plot_scatter_with_job_length(xaxis_pts, yaxis_pts, z=coloraxis_pts, xlabel=xtitle, ylabel=ytitle, output=output, log=log, title="Queueing Delay over Completion Time")


def graph_queueing_delay_over_planned(options: argparse.Namespace):
    # queueing_delay = "batch_instance_analysis/queueing_delays_merged.csv"
    num_points = options.num_points or 1_000_000
    dag = options.dag or True
    if dag:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_dag_requested_{num_points}.csv")
        queueing_delay = "batch_instance_analysis/queueing_delays_dag.csv"
    else:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_requested_{num_points}.csv")
        queueing_delay = "batch_instance_analysis/queueing_delays.csv"
    output_dir = "queueing_delay_graphs"
    os.makedirs(output_dir, exist_ok=True)
    num_lines = 1_228_679_841
    n_bins = 100

    if os.path.exists(cache_file) and not options.overwrite:
        df = pd.read_csv(cache_file)
        timestamps = df["timestamp"]
        qdelay_pts = df["queueing_delay"]
        completion_times = df["completion_time"]
        job_lengths = df["job_length"]
        plan_cpus = df["plan_cpu"]
        plan_mems = df["plan_mem"]
        cpu_avgs = df["cpu_avg"]
        cpu_maxes = df["cpu_max"]
        mem_avgs = df["mem_avg"]
        mem_maxes = df["mem_max"]
        del df
    else:
        if os.path.exists(cache_file):
            os.remove(cache_file)
        qdelay_pts = []
        timestamps = []
        completion_times = []
        job_lengths = []
        plan_cpus = []
        plan_mems = []
        cpu_avgs = []
        cpu_maxes = []
        mem_avgs = []
        mem_maxes = []

        chunksize = 100_000_000
        max_iter = num_lines // chunksize + 1
        size = int(num_points / max_iter)
        requested_info = pd.read_csv("alibaba-2018.csv", names=['task_name', 'instance_num', 'job_name', 'task_type', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem'])

        logger.info(f"looping {max_iter} times and selecting {size} data pts each time")
        start = time.time()
        for j, data in enumerate(
                pd.read_csv(queueing_delay, engine="c", chunksize=chunksize)):
            merged_df = requested_info.merge(data, how='inner', on=['job_name', 'task_name']).dropna()
            sampled_data = merged_df.sample(size, replace=False)
            qdelay_pts.extend(sampled_data["queueing_delay"])
            timestamps.extend(sampled_data["timestamp"])
            completion_times.extend(sampled_data["completion_time"])
            job_lengths.extend(sampled_data["job_length"])
            plan_cpus.extend(sampled_data["plan_cpu"])
            plan_mems.extend(sampled_data["plan_mem"])
            cpu_avgs.extend(sampled_data["cpu_avg"])
            cpu_maxes.extend(sampled_data["cpu_max"])
            mem_avgs.extend(sampled_data["mem_avg"])
            mem_maxes.extend(sampled_data["mem_max"])
            logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
            sampled_data.to_csv(cache_file, header=not os.path.exists(cache_file), mode="a")

    def remove_negative(l):
        return all(x >= 0 for x in l)

    def remove_inf(l):
        return all(x >= np.inf for x in l)

    # timestamps, qdelay_pts = (list(t) for t in zip(*sorted(filter(remove_negative, zip(timestamps, qdelay_pts)))))
    timestamps, completion_times, qdelay_pts, job_lengths, cpu_avgs, cpu_maxes, mem_avgs, mem_maxes, plan_cpus, plan_mems = (list(t) for t in zip(*sorted(
        filter(remove_negative, zip(timestamps, completion_times, qdelay_pts, job_lengths, cpu_avgs, cpu_maxes, mem_avgs, mem_maxes, plan_cpus, plan_mems)))))
    output = os.path.join(output_dir, "queueing_delay_over_plan_cpu.png")
    plot_scatter_with_job_length(plan_cpus, qdelay_pts, job_lengths, xlabel="plan_cpu", ylabel="queueing delay", output=output)
    output = os.path.join(output_dir, "queueing_delay_over_plan_mem.png")
    plot_scatter_with_job_length(plan_mems, qdelay_pts, job_lengths, xlabel="plan_mem", ylabel="queueing delay", output=output)
    output = os.path.join(output_dir, "queueing_delay_over_oversubscription_mem_avg.png")
    plot_scatter_with_job_length(np.array(plan_mems) / np.array(mem_avgs), qdelay_pts, job_lengths, xlabel="oversubscription mem avg", ylabel="queueing delay", output=output)
    output = os.path.join(output_dir, "queueing_delay_over_oversubscription_mem_max.png")
    plot_scatter_with_job_length(np.array(plan_mems) / np.array(mem_maxes), qdelay_pts, job_lengths, xlabel="oversubscription mem max", ylabel="queueing delay", output=output)
    output = os.path.join(output_dir, "queueing_delay_over_oversubscription_cpu_avg.png")
    plot_scatter_with_job_length(np.array(plan_cpus) / np.array(cpu_avgs), qdelay_pts, job_lengths, xlabel="oversubscription cpu avg", ylabel="queueing delay", output=output)
    output = os.path.join(output_dir, "queueing_delay_over_oversubscription_cpu_max.png")
    plot_scatter_with_job_length(np.array(plan_cpus) / np.array(cpu_maxes), qdelay_pts, job_lengths, xlabel="oversubscription cpu max", ylabel="queueing delay", output=output)


def plot_scatter_with_job_length(x, y, z, xlabel, ylabel, output, log=True, title=None):
    plt.figure(figsize=(12, 8))
    scatter_plot = plt.scatter(x, y, 0.5, c=z, cmap='viridis', norm='log' if log else 'linear')
    cbar = plt.colorbar(scatter_plot)
    cbar.set_label("Execution Time in seconds", fontsize=24)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.grid(True, which="both", ls="--")
    if log:
        plt.xscale('log')
        plt.yscale('log')
    if title:
        plt.title(title, fontsize=24)
    plt.savefig(output)


# graph all: python graph_queueing_delay.py -o graph_queueing_delay_over_completion && python graph_queueing_delay.py -o graph_queueing_delay_over_cpu_max && python
# graph_queueing_delay.py -o graph_queueing_delay_over_cpu_avg && python graph_queueing_delay.py -o graph_queueing_delay_over_mem_max && python graph_queueing_delay.py -o
# graph_queueing_delay_over_mem_avg && python graph_queueing_delay.py -o graph_queueing_delay_over_planned
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Run a specified operation.")
    parser.add_argument("--operation", "-o", type=str, help="Name of the operation (function) to execute.")
    parser.add_argument("--overwrite", action="store_true", default=-False)
    parser.add_argument("--num-points", "-n", type=int)
    parser.add_argument("--dag", type=bool, default=True)
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


# https://stackoverflow.com/questions/55057957/an-attempt-has-been-made-to-start-a-new-process-before-the-current-process-has-f
if __name__ == "__main__":
    main()

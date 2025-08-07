import argparse
import functools
import math
import os
import sys
import datetime as dt
from collections import defaultdict
from functools import wraps

from importlib import import_module
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import time

import logging

from matplotlib import pyplot as plt
from pyarrow import csv

from pytz import timezone

from scipy.stats import gaussian_kde

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


def calcProcessTime(starttime, cur_iter, max_iter):
    telapsed = time.time() - starttime
    testimated = (telapsed / cur_iter) * (max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).astimezone(timezone("US/Pacific")).strftime("%H:%M:%S")  # in time

    lefttime = testimated - telapsed  # in seconds

    return int(telapsed), int(lefttime), finishtime


def get_unique_status(options: argparse.Namespace):
    """
    :param options: namespace object
    """

    if type == "avg":
        mem_type = "mem_avg"
        cpu_type = "cpu_avg"
    else:
        mem_type = "mem_max"
        cpu_type = "cpu_max"
    data_path = options.trace_file or "../alibaba-trace/batch_instance.csv"

    # reading all at once takes too much mem, batch_instance.csv is 113GB
    chunksize = 10_000_000
    # data = pd.read_csv(data_path, engine="c", names=names, low_memory=True)  # takes too much memory

    start = time.time()
    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1
    lines_read = 0

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    unique_status = defaultdict(int)
    for j, data in enumerate(pd.read_csv(data_path, names=names, engine="pyarrow", chunksize=chunksize, header=None)):
        # do the normalization later, keep/sum the raw values
        # data.loc[:, [mem_type]] = data.loc[:, [mem_type]].div(4023)
        # data.loc[:, [cpu_type]] = data.loc[:, [cpu_type]].div(4023 * 96)

        data['Core'] = pd.to_numeric(data[cpu_type], errors='coerce')
        data['Memory'] = pd.to_numeric(data[mem_type], errors='coerce')
        data['Start Time'] = pd.to_numeric(data['start_time'], errors='coerce')
        data['End Time'] = pd.to_numeric(data['end_time'], errors='coerce')
        data = data.dropna(subset=[cpu_type, mem_type])
        # data = data[data['status'] == 'Terminated'].copy()

        data_np = data.to_numpy()

        for row in data_np:
            status = row[4]
            unique_status[status] += 1
        lines_read += chunksize

        logger.info(f"{chunksize * (j + 1) / num_lines * 100:.2f}% complete")
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    end = time.time()
    logger.info(f"took {end - start}")
    print(unique_status)


def instances_of_job_task_sort(options: argparse.Namespace):
    output_directory = options.output

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())
    lookup_job_name = options.lookup_job
    lookup_task_name = options.lookup_task
    assert options.lookup_job is not None
    assert options.lookup_task is not None

    data_file = os.path.join(output_directory, f"lookup_instances_of_job_and_task_{lookup_job_name}_{lookup_task_name}.csv")
    df = pd.read_csv(data_file, names=names)
    df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")
    df['cpu_avg'] = pd.to_numeric(df["cpu_avg"], errors='coerce')
    df['mem_avg'] = pd.to_numeric(df["mem_avg"], errors='coerce')
    df['cpu_max'] = pd.to_numeric(df["cpu_max"], errors='coerce')
    df['mem_max'] = pd.to_numeric(df["mem_max"], errors='coerce')
    df.sort_values("start_time", inplace=True)
    new_data_file = os.path.join(output_directory, f"sorted_lookup_instances_of_job_and_task_{lookup_job_name}_{lookup_task_name}.csv")
    df.to_csv(new_data_file, header=False, index=False)
    cpu_avg_std = np.std(df["cpu_avg"])
    cpu_max_std = np.std(df["cpu_max"])
    mem_avg_std = np.std(df["mem_avg"])
    mem_max_std = np.std(df["mem_max"])

    print(f"cpu_avg std: {cpu_avg_std}")
    print(f"cpu_max std: {cpu_max_std}")
    print(f"mem_avg std: {mem_avg_std}")
    print(f"mem_max std: {mem_max_std}")

    print(f"cpu_avg min-max value: [{df['cpu_avg'].min()}, {df['cpu_avg'].max()}]")
    print(f"cpu_max min-max value: [{df['cpu_max'].min()}, {df['cpu_max'].max()}]")
    print(f"mem_avg min-max value: [{df['mem_avg'].min()}, {df['mem_avg'].max()}]")
    print(f"mem_max min-max value: [{df['mem_max'].min()}, {df['mem_max'].max()}]")

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    n_bins = 100
    axs[0][0].hist(df["cpu_avg"], bins=n_bins)
    axs[0][0].set_title("cpu_avg distribution")
    axs[0][1].hist(df["cpu_max"], bins=n_bins)
    axs[0][1].set_title("cpu_max distribution")
    axs[1][0].hist(df["mem_avg"], bins=n_bins)
    axs[1][0].set_title("mem_avg distribution")
    axs[1][1].hist(df["mem_max"], bins=n_bins)
    axs[1][1].set_title("mem_max distribution")

    distribution_plot = os.path.join(output_directory, f"distribution_instances_of_job_and_task_{lookup_job_name}_{lookup_task_name}.png")
    plt.show()
    plt.savefig(distribution_plot)
    plt.close()


def get_instances_of_job_task(options: argparse.Namespace):
    data_path = options.trace_file or "../alibaba-trace/batch_instance.csv"
    output_directory = options.output
    chunksize = 50_000_000

    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())
    lookup_job_name = options.lookup_job
    lookup_task_name = options.lookup_task
    assert options.lookup_job is not None
    assert options.lookup_task is not None
    found: list[pd.DataFrame] = list()
    start = time.time()
    for j, data in enumerate(pd.read_csv(data_path, names=names, engine="c", chunksize=chunksize, header=None)):
        data['cpu_avg'] = pd.to_numeric(data["cpu_avg"], errors='coerce')
        data['mem_avg'] = pd.to_numeric(data["mem_avg"], errors='coerce')
        data['cpu_max'] = pd.to_numeric(data["cpu_max"], errors='coerce')
        data['mem_max'] = pd.to_numeric(data["mem_max"], errors='coerce')
        # data['start_time'] = pd.to_numeric(data['start_time'], errors='coerce')
        # data['end_time'] = pd.to_numeric(data['end_time'], errors='coerce')
        data = data.dropna(subset=["cpu_avg", "cpu_max", "mem_avg", "mem_max"])
        data = data[data['status'] == 'Terminated'].copy()

        # Job Name        j_2910082
        # Task Name              M1
        # Job Name                                j_3062477
        # Task Name       task_LTI2NTc5ODQ0OTQxMTIyNDk5MTY=
        lookup = data.loc[(data["task_name"] == lookup_task_name) & (data["job_name"] == lookup_job_name)]
        if not lookup.empty:
            found.append(lookup)

        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    data_file = os.path.join(output_directory, f"lookup_instances_of_job_and_task_{lookup_job_name}_{lookup_task_name}.csv")

    new_df = pd.concat(found)
    new_df.to_csv(data_file, mode="w", header=False, index=False)
    end = time.time()
    logger.info(f"extracted to {data_file}, took {end - start}")


def get_batch_instance_to_task_usage(options: argparse.Namespace):
    """
    Convert batch instance information into per task information

    Will take overall statistics of CPU/Mem avg/max over all instances of a task and convert
    them (average) into singular CPU/Mem avg/max values per task
    :param options: namespace object
    """
    data_path = options.trace_file or "../alibaba-trace/batch_instance.csv"

    # input_file_name = get_clean_filename(data_path)
    output_directory = options.output
    # file = get_plot_filename(input_file_name, output_directory, options.output)
    conversion_alg = options.instance_to_task_estimation
    data_file = os.path.join(output_directory, f"batch_instance_to_task_data_{conversion_alg}.csv")

    if os.path.exists(data_file) and not options.overwrite:
        print(f"File at {data_file} already exists.")
        return

    # reading all at once takes too much mem, batch_instance.csv is 113GB
    chunksize = 10_000_000
    # data = pd.read_csv(data_path, engine="c", names=names, low_memory=True)  # takes too much memory

    start = time.time()
    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1
    lines_read = 0

    if os.path.exists(data_file):
        os.remove(data_file)

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    task_to_info = dict()

    for j, data in enumerate(pd.read_csv(data_path, names=names, engine="c", chunksize=chunksize, header=None)):
        data['cpu_avg'] = pd.to_numeric(data["cpu_avg"], errors='coerce')
        data['mem_avg'] = pd.to_numeric(data["mem_avg"], errors='coerce')
        data['cpu_max'] = pd.to_numeric(data["cpu_max"], errors='coerce')
        data['mem_max'] = pd.to_numeric(data["mem_max"], errors='coerce')
        # data['start_time'] = pd.to_numeric(data['start_time'], errors='coerce')
        # data['end_time'] = pd.to_numeric(data['end_time'], errors='coerce')
        data = data.dropna(subset=["cpu_avg", "cpu_max", "mem_avg", "mem_max"])
        data = data[data['status'] == 'Terminated'].copy()

        # dont need to sort by timestamp
        data_np = data.to_numpy()
        for row in data_np:
            instance_name, task_name, job_name, task_type, status, start_time, end_time, machine_id, seq_no, total_seq_no, cpu_avg, cpu_max, mem_avg, mem_max = row
            if task_to_info.get((job_name, task_name), None) is None:
                task_to_info[(job_name, task_name)] = {"cpu_avg": cpu_avg, "cpu_max": cpu_max, "mem_avg": mem_avg, "mem_max": mem_max, "count": 1}
            else:
                info = task_to_info[(job_name, task_name)]
                prev_count = info["count"]
                old_cpu_avg = info["cpu_avg"]
                old_cpu_max = info["cpu_max"]
                old_mem_avg = info["mem_avg"]
                old_mem_max = info["mem_max"]
                new_count = prev_count + 1
                # Get the average cpu_avg, cpu_max, etc over time per task
                if conversion_alg == "avg":
                    new_cpu_avg = (old_cpu_avg * prev_count + cpu_avg) / new_count
                    new_cpu_max = (old_cpu_max * prev_count + cpu_max) / new_count
                    new_mem_avg = (old_mem_avg * prev_count + mem_avg) / new_count
                    new_mem_max = (old_mem_max * prev_count + mem_max) / new_count
                elif conversion_alg == "max":
                    new_cpu_avg = cpu_avg if cpu_avg > old_cpu_avg else old_cpu_avg
                    new_cpu_max = cpu_max if cpu_max > old_cpu_max else old_cpu_max
                    new_mem_avg = mem_avg if mem_avg > old_mem_avg else old_mem_avg
                    new_mem_max = mem_max if mem_max > old_mem_max else old_mem_max
                else:
                    raise RuntimeError(f"Unknown operation {conversion_alg}")
                task_to_info[(job_name, task_name)] = {"cpu_avg": new_cpu_avg, "cpu_max": new_cpu_max, "mem_avg": new_mem_avg, "mem_max": new_mem_max, "count": new_count}
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    del data
    coalesced_data = {"job_name": list(), "task_name": list(), "cpu_avg": list(), "cpu_max": list(), "mem_avg": list(), "mem_max": list()}
    for name, info in task_to_info.items():
        coalesced_data["job_name"].append(name[0])
        coalesced_data["task_name"].append(name[1])
        # note: not normalized, 100 = 1cpu
        coalesced_data["cpu_avg"].append(info["cpu_avg"])
        coalesced_data["cpu_max"].append(info["cpu_max"])
        coalesced_data["mem_avg"].append(info["mem_avg"])
        coalesced_data["mem_max"].append(info["mem_max"])
    del task_to_info
    new_df = pd.DataFrame.from_dict(coalesced_data)
    new_df.to_csv(data_file, mode="w", header=False, index=False)
    end = time.time()
    logger.info(f"extracted to {data_file}, took {end - start}")


def compare_batch_instance_to_task(options: argparse.Namespace):
    """
    Take the converted task data (which holds cpu+mem avg+max values per task)
    and compare them to the requested values

    get_batch_instance_to_task_usage() must be run before this
    """
    output_directory = options.output
    conversion_alg = options.instance_to_task_estimation
    instance_to_task_file = os.path.join(output_directory, f"batch_instance_to_task_data_{conversion_alg}.csv")

    batch_task_file = "alibaba-2018.csv"
    names = ["job_name", "task_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max"]
    requested_task_data = pd.read_csv(batch_task_file, names=['task_name', 'instance_num', 'job_name', 'task_type', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem'])
    instance_to_task_data = pd.read_csv(instance_to_task_file, names=names)
    requested_task_data.set_index(["job_name", "task_name"], inplace=True)
    instance_to_task_data.set_index(["job_name", "task_name"], inplace=True)

    reindexed_instance_to_task_data = instance_to_task_data.reindex(requested_task_data.index)

    # We don't want the raw comparison value but the percentage difference instead
    # This should be the same as percent error in statistics
    # (real - requested) / requested
    # Note, we keep the same normalization scale in batch_instance_to_task_data.csv as the requested trace data (ex 100=1cpu)
    # This way we don't have to do any preprocessing or conversion at this step
    task_difference_cpu_avg = (reindexed_instance_to_task_data["cpu_avg"] - requested_task_data["plan_cpu"]) / requested_task_data["plan_cpu"]
    result_df_cpu_avg = task_difference_cpu_avg.reset_index().rename(columns={0: 'difference_cpu_avg'})
    task_difference_cpu_max = (reindexed_instance_to_task_data["cpu_max"] - requested_task_data["plan_cpu"]) / requested_task_data["plan_cpu"]
    result_df_cpu_max = task_difference_cpu_max.reset_index().rename(columns={0: 'difference_cpu_max'})
    task_difference_mem_avg = (reindexed_instance_to_task_data["mem_avg"] - requested_task_data["plan_mem"]) / requested_task_data["plan_mem"]
    result_df_mem_avg = task_difference_mem_avg.reset_index().rename(columns={0: 'difference_mem_avg'})
    task_difference_mem_max = (reindexed_instance_to_task_data["mem_max"] - requested_task_data["plan_mem"]) / requested_task_data["plan_mem"]
    result_df_mem_max = task_difference_mem_max.reset_index().rename(columns={0: 'difference_mem_max'})

    # Throw the percent difference/error all in a singular csv file
    result_df = pd.concat([result_df_cpu_avg, result_df_cpu_max, result_df_mem_avg, result_df_mem_max], axis=1)
    result_df.drop(columns=["job_name", "task_name"], inplace=True)
    # result_df["plan_cpu"] = requested_task_data["plan_cpu"].values
    # result_df["cpu_max"] = reindexed_instance_to_task_data["cpu_max"].values

    data_file = os.path.join(output_directory, f"batch_task_difference_{conversion_alg}.csv")
    result_df.to_csv(data_file, index=False, mode="w")


def plot_batch_instance_to_task_cdf(options: argparse.Namespace):
    output_directory = options.output
    conversion_alg = options.instance_to_task_estimation
    input_data_file = os.path.join(output_directory, f"batch_task_difference_{conversion_alg}.csv")
    df = pd.read_csv(input_data_file)
    df = df[["difference_cpu_avg", "difference_cpu_max", "difference_mem_avg", "difference_mem_max"]]

    difference_cpu_avg = df[f"difference_cpu_avg"].dropna()
    difference_cpu_max = df[f"difference_cpu_max"].dropna()
    difference_mem_avg = df[f"difference_mem_avg"].dropna()
    difference_mem_max = df[f"difference_mem_max"].dropna()

    all_differences = {"cpu_avg": difference_cpu_avg, "cpu_max": difference_cpu_max, "mem_avg": difference_mem_avg, "mem_max": difference_mem_max}

    for typ, differences in all_differences.items():
        start = time.time()
        plot_filename = os.path.join(output_directory, f"batch_instance_{conversion_alg}_to_task_difference_{typ}_cdf.png")
        x, y = calculate_cdf(differences)

        plt.figure(figsize=(10, 10))
        fig, ax1 = plt.subplots()

        ax1.plot(x, y)
        ax1.set_title(f"CDF {typ}")
        ax1.set_ylabel("CDF %")
        ax1.set_xlabel(f"% error ({typ})")
        plt.savefig(plot_filename)
        print(f"Created plot {plot_filename}")
        plt.clf()
        end = time.time()
        logger.info(f"Plotted in {end - start} seconds")


def plot_batch_instance_to_task_pdf(options: argparse.Namespace):
    def get_pdf(differences):
        # use KDE for PDF
        # https://stackoverflow.com/questions/45464924/python-calculating-pdf-from-a-numpy-array-distribution
        # gaussian smoothing will be used
        kde = gaussian_kde(differences)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.integrate_box_1d.html
        # print(kde.integrate_box_1d(-np.inf, np.inf))
        x_pts = np.linspace(differences.min(), differences.max(), num=10000)
        evaluated = kde.evaluate(x_pts)
        return x_pts, evaluated

    output_directory = options.output
    conversion_alg = options.instance_to_task_estimation
    input_data_file = os.path.join(output_directory, f"batch_task_difference_{conversion_alg}.csv")

    start = time.time()
    df = pd.read_csv(input_data_file)

    df = df[["difference_cpu_avg", "difference_cpu_max", "difference_mem_avg", "difference_mem_max"]]
    end = time.time()
    logger.info(f"Read CSV data in {end - start} seconds")

    typ = "cpu_avg"
    difference_cpu_avg = df[f"difference_{typ}"].dropna()

    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots()
    plot_filename = os.path.join(output_directory, f"batch_instance_{conversion_alg}_to_task_difference_{typ}_pdf.png")
    x, y = get_pdf(difference_cpu_avg)
    ax1.plot(x, y)
    ax1.set_title(f"PDF {typ}")
    ax1.set_ylabel("PDF %")
    ax1.set_xlabel(f"% error {typ}")
    plt.savefig(plot_filename)
    print(f"Created plot {plot_filename}")
    end = time.time()
    logger.info(f"Plotted in {end - start} seconds")

    plt.clf()

    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots()
    typ = "cpu_max"
    difference_cpu_max = df[f"difference_{typ}"].dropna()
    plot_filename = os.path.join(output_directory, f"batch_instance_{conversion_alg}_to_task_difference_{typ}_pdf.png")
    x, y = get_pdf(difference_cpu_max)
    ax1.plot(x, y)
    ax1.set_title(f"PDF {typ}")
    ax1.set_ylabel("PDF %")
    ax1.set_xlabel(f"% error ({typ})")
    plt.savefig(plot_filename)
    print(f"Created plot {plot_filename}")
    plt.clf()
    end = time.time()
    logger.info(f"Saved in {end - start} seconds")

    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots()
    typ = "mem_avg"
    difference_mem_avg = df[f"difference_{typ}"].dropna()
    plot_filename = os.path.join(output_directory, f"batch_instance_{conversion_alg}_to_task_difference_{typ}_pdf.png")
    x, y = get_pdf(difference_mem_avg)
    ax1.plot(x, y)
    ax1.set_title(f"PDF {typ}")
    ax1.set_ylabel("PDF %")
    ax1.set_xlabel(f"% error {typ}")
    plt.savefig(plot_filename)
    print(f"Created plot {plot_filename}")
    plt.clf()
    end = time.time()
    logger.info(f"Saved in {end - start} seconds")

    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots()
    typ = "mem_max"
    difference_mem_max = df[f"difference_{typ}"].dropna()
    plot_filename = os.path.join(output_directory, f"batch_instance_{conversion_alg}_to_task_difference_{typ}_pdf.png")
    x, y = get_pdf(difference_mem_max)
    ax1.plot(x, y)
    ax1.set_title(f"PDF {typ}")
    ax1.set_ylabel("PDF %")
    ax1.set_xlabel(f"% error {typ}")
    plt.savefig(plot_filename)
    print(f"Created plot {plot_filename}")
    plt.clf()
    end = time.time()
    logger.info(f"Saved in {end - start} seconds")
    return

    percents = np.array(list(range(0, 1001, 1))) / 1000
    percentiles_cpu_avg = difference_cpu_avg.quantile(percents)
    percentiles_cpu_max = difference_cpu_max.quantile(percents)

    percentiles_mem_avg = difference_mem_avg.quantile(percents)
    percentiles_mem_max = difference_mem_max.quantile(percents)

    plt.figure(figsize=(10, 10))
    fig, axs = plt.subplots(2, 2)

    axs[0][0].set_title('Count/percentile difference\n(cpu_avg)')
    axs[0][0].set_xlabel("Percentile of difference")
    axs[0][0].set_ylabel("Difference (cpu_avg) counts")
    axs[0][0].plot(percents, percentiles_cpu_avg)
    axs[0][0].tick_params(axis='y')

    axs[0][1].set_title('Count/percentile difference\n(cpu_max)')
    axs[0][1].set_xlabel("Percentile of difference")
    axs[0][1].set_ylabel('Difference (cpu_max) counts')  # we already handled the x-label with ax1
    axs[0][1].plot(percents, percentiles_cpu_max)
    axs[0][1].tick_params(axis='y')

    axs[1][0].set_title('Count/percentile difference\n(mem_avg)')
    axs[1][0].set_xlabel("Percentile of difference")
    axs[1][0].set_ylabel("Difference (mem_avg) counts")
    axs[1][0].plot(percents, percentiles_mem_avg)
    axs[1][0].tick_params(axis='y')

    axs[1][1].set_title('Count/percentile difference\n(mem_max)')
    axs[1][1].set_xlabel("Percentile of difference")
    axs[1][1].set_ylabel('Difference (mem_max) counts')  # we already handled the x-label with ax3
    axs[1][1].plot(percents, percentiles_mem_max)
    axs[1][1].tick_params(axis='y')

    fig.tight_layout()
    plt.tight_layout()
    plot_filename = os.path.join(output_directory, f"batch_task_difference_percentiles.png")
    plt.subplots_adjust(hspace=1)
    plt.savefig(plot_filename)
    plt.close()


def create_plot_one(timestamps: List[float], cores: List[float], input_file_name: str, output_directory: str, output: Optional[str] = None,
                    write_config: bool = True,
                    multiplier_data: Optional[Tuple[float, float, float]] = None, cores_label=None, mem_label=None, time_label=None):
    if output is not None:
        output_filename = output
    else:
        output_filename = get_plot_filename(input_file_name, output_directory, output)

    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Requested Cores', color=color)
    ax1.set_xlabel(time_label or 'Timestamp (seconds)')
    ax1.set_ylabel(cores_label or 'CPU Cores', color=color)
    ax1.plot(timestamps, cores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    plt.savefig(output_filename)
    plt.close()
    # ax2 = ax1.twinx()
    color = 'tab:blue'

    print(f"Plot saved to: {output_filename}")


def extract_batch_instance_type(options: argparse.Namespace, type="avg"):
    """
    :param options: namespace object
    """

    if type == "avg":
        mem_type = "mem_avg"
        cpu_type = "cpu_avg"
    else:
        mem_type = "mem_max"
        cpu_type = "cpu_max"
    data_path = options.trace_file or "../alibaba-trace/batch_instance.csv"

    input_file_name = get_clean_filename(data_path)
    output_directory = options.output
    file = get_plot_filename(input_file_name, output_directory, options.output)
    graph_data_file = os.path.join(output_directory, f"batch_instance_{type}_data.csv")

    if os.path.exists(graph_data_file) and not options.overwrite:
        print(f"File at {graph_data_file} already exists.")
        return

    # reading all at once takes too much mem, batch_instance.csv is 113GB
    chunksize = 10_000_000
    # data = pd.read_csv(data_path, engine="c", names=names, low_memory=True)  # takes too much memory

    start = time.time()
    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1
    lines_read = 0

    if os.path.exists(graph_data_file):
        os.remove(graph_data_file)

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    current_cores: float = 0.0
    current_rss: float = 0.0
    for j, data in enumerate(pd.read_csv(data_path, names=names, engine="c", chunksize=chunksize, header=None)):
        # do the normalization later, keep/sum the raw values
        # data.loc[:, [mem_type]] = data.loc[:, [mem_type]].div(4023)
        # data.loc[:, [cpu_type]] = data.loc[:, [cpu_type]].div(4023 * 96)

        data['Core'] = pd.to_numeric(data[cpu_type], errors='coerce')
        data['Memory'] = pd.to_numeric(data[mem_type], errors='coerce')
        data['Memory'] = pd.to_numeric(data['start_time'], errors='coerce')
        data['Memory'] = pd.to_numeric(data['end_time'], errors='coerce')
        data = data.dropna(subset=[cpu_type, mem_type])
        data = data[data['status'] == 'Terminated'].copy()

        # Process the data for utilization calculation
        start_data = data[['start_time', cpu_type, mem_type]].copy()
        start_data.columns = ['timestamp', 'cores', 'rss']
        start_data['event_type'] = 'start'

        end_data = data[['end_time', cpu_type, mem_type]].copy()
        end_data.columns = ['timestamp', 'cores', 'rss']
        end_data['event_type'] = 'end'

        combined_data = pd.concat([start_data, end_data]).sort_values('timestamp')
        combined_data_ndarray = combined_data.to_numpy()

        cores: List[float] = []
        rss: List[float] = []
        timestamps = []
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

        # since it's too big, dump into a file for later
        parsed_df = pd.DataFrame({"timestamp": timestamps, "cores": cores, "rss": rss})
        min_timestamp = min(parsed_df["timestamp"])
        max_timestamp = max(parsed_df["timestamp"])
        step = 30  # every half minute
        s = pd.Series(list(range(min_timestamp, max_timestamp, step)), name="times")
        trimmed_df = pd.merge_asof(s, parsed_df, left_on="times", right_on="timestamp", direction="forward")  # look for the first match that is above the threshold
        trimmed_df.set_index("timestamp", inplace=True)
        trimmed_df.to_csv(graph_data_file, mode="a", header=False)

        logger.info(f"{chunksize * (j + 1) / num_lines * 100:.2f}% complete")
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    end = time.time()
    logger.info(f"extracted to {graph_data_file}, took {end - start}")


def plot_batch_instance_type(options: argparse.Namespace, type="avg"):
    output_directory = options.output

    plot_filename = os.path.join(output_directory, f"batch_instance_{type}.png")
    if os.path.exists(plot_filename) and not options.overwrite:
        print(f"File at {plot_filename} exists!")
        return

    graph_data_file = options.infile or os.path.join(output_directory, f"batch_instance_{type}_data.csv")
    # since we parsed the raw data, normalize here
    names = ["timestamp", "times", "cores", "rss"]
    data = pd.read_csv(graph_data_file, header=None, names=names)

    data['cores'] = pd.to_numeric(data['cores'], errors='coerce')
    data['rss'] = pd.to_numeric(data['rss'], errors='coerce')

    # data = data.nlargest(1000, columns="rss")
    # normalize so 1=1 cpu core and 1=1 mem normalized unit
    data.loc[:, ["rss"]] = data.loc[:, ["rss"]].div(100)
    data.loc[:, ["cores"]] = data.loc[:, ["cores"]].div(100)
    if options.norm:
        data.loc[:, ["rss"]] = data.loc[:, ["rss"]].div(4023)
        data.loc[:, ["cores"]] = data.loc[:, ["cores"]].div(4023 * 96)
        cores_label = "CPU usage (total)"
        mem_label = "Mem usage (total)"
    else:
        cores_label = "CPU (total)"
        mem_label = "Mem (total)"

    data.sort_values('timestamp', inplace=True)

    timestamps = data["timestamp"].tolist()
    cores = data["cores"].tolist()
    rss = data["rss"].tolist()

    print(f"Plotting {len(timestamps)} points")

    create_plot(timestamps, cores, rss, input_file_name=os.path.basename(graph_data_file), output_directory=output_directory, write_config=False,
                cores_label=cores_label, mem_label=mem_label, output=plot_filename)


def extract_machine_usage(options: argparse.Namespace):
    """
    Extract out the utilization of the cluster (remaining total resources over the entire cluster and the max to fit in one machine
    """
    data_path = options.trace_file
    if data_path == "../alibaba-trace/batch_instance.csv":
        data_path = "../alibaba-trace/machine_usage.csv"

    num_machines = 4023

    output_directory = "machine_instance_analysis"
    output_file = os.path.join(output_directory, "remaining_machine_usage.csv")
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if os.path.exists(output_file) and options.overwrite is False:
        raise RuntimeError(f"File at {output_file} exists")

    if os.path.exists(output_file):
        os.remove(output_file)

    chunksize = 10_000_000
    start = time.time()
    num_lines = 246_934_820
    max_iter = num_lines // chunksize + 1
    lines_read = 0
    names = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent", "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    recording_time_stamp = None

    core_usage = 0
    mem_usage = 0
    # largest number of available resources local to one machine instead of the overall cluster
    min_machine_cpu_util = 0
    min_machine_mem_util = 0

    data = pd.read_csv(data_path, names=names, engine="c", header=None)
    remaining_core = []
    remaining_memory = []
    per_machine_max_remaining_core = []
    per_machine_max_remaining_memory = []
    timestamps = []
    # borrowed code from machine_analyze.py
    data.sort_values("time_stamp", inplace=True)
    data_ndarray = data.to_numpy()
    recording_time_stamp = recording_time_stamp or data_ndarray[0][1]

    for row in data_ndarray:
        time_stamp = int(row[1])
        cpu_util_percent = float(row[2])
        mem_util_percent = float(row[3])

        if time_stamp == recording_time_stamp:
            core_usage += cpu_util_percent
            mem_usage += mem_util_percent
            if cpu_util_percent > min_machine_cpu_util:
                min_machine_cpu_util = cpu_util_percent
            if mem_util_percent > min_machine_mem_util:
                min_machine_mem_util = mem_util_percent
        else:
            timestamps.append(recording_time_stamp)
            remaining_core.append(1 - (core_usage / num_machines / 100))  # make 1=1cpu core
            remaining_memory.append(1 - (mem_usage / num_machines / 100))  # make 1=1 normalized memory unit per machine
            per_machine_max_remaining_core.append(min_machine_cpu_util)
            per_machine_max_remaining_memory.append(min_machine_mem_util)
            core_usage = cpu_util_percent
            mem_usage = mem_util_percent
            recording_time_stamp = time_stamp
            min_machine_cpu_util = 0
            min_machine_mem_util = 0
    if core_usage > 0 or mem_usage > 0:
        timestamps.append(recording_time_stamp)
        remaining_core.append(1 - (core_usage / num_machines / 100))
        remaining_memory.append(1 - (mem_usage / num_machines / 100))
        per_machine_max_remaining_core.append(min_machine_cpu_util)
        per_machine_max_remaining_memory.append(min_machine_mem_util)
    output_df = pd.DataFrame({"timestamp": timestamps, "remaining_cpu_cluster": remaining_core,
                              "remaining_mem_cluster": remaining_memory, "remaining_cpu_machine": per_machine_max_remaining_core,
                              "remaining_mem_machine": per_machine_max_remaining_memory}, columns=["timestamp", "remaining_cpu_cluster", "remaining_mem_cluster",
                                                                                                   "remaining_cpu_machine", "remaining_mem_machine"])
    output_df.to_csv(output_file, header=False, index=False)

    end = time.time()

    print(f"Completed extraction to {output_file} in {end - start} seconds")


def plot_remaining_machine_usage(options: argparse.Namespace):
    columns = ["timestamp", "remaining_cpu_cluster", "remaining_mem_cluster",
               "remaining_cpu_machine", "remaining_mem_machine"]

    machine_instance_directory = "machine_instance_analysis"
    input_file = os.path.join(machine_instance_directory, "remaining_machine_usage.csv")
    df = pd.read_csv(input_file, names=columns, header=None)
    create_plot(df["timestamp"], df["remaining_cpu_cluster"], df["remaining_mem_cluster"], input_file_name=os.path.basename(input_file),
                output_directory=machine_instance_directory, write_config=False,
                cores_label="Remaining CPU", mem_label="Remaining Mem (norm)", output=os.path.join(machine_instance_directory, "remaining_machine_usage_cluster.png"))

    create_plot(df["timestamp"], df["remaining_cpu_machine"], df["remaining_mem_machine"], input_file_name=os.path.basename(input_file),
                output_directory=machine_instance_directory, write_config=False,
                cores_label="Remaining CPU", mem_label="Remaining Mem (norm)", output=os.path.join(machine_instance_directory, "remaining_machine_usage_machine.png"))


def test_start_end_timestamp(options: argparse.Namespace):
    data_path = options.trace_file or "../alibaba-trace/batch_instance_sort.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())
    chunksize = 100_000
    for j, data in enumerate(pd.read_csv(data_path, names=names, engine="c", chunksize=chunksize, header=None)):
        data_ndarray = data.to_numpy()
        for row in data_ndarray:
            status = row[4]
            start_time = int(row[5])
            end_time = int(row[6])
            if end_time < start_time and status == "Terminated":
                logger.info(f"{row}")


def reciprocal_bloat(options: argparse.Namespace):
    data_path = options.trace_file or "../alibaba-trace/batch_instance_sort.csv"
    batch_task_request = "alibaba-2018.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    if not os.path.exists("oversubscription_analysis"):
        os.mkdir("oversubscription_analysis")
    output_file = os.path.join("oversubscription_analysis", "running_usage.csv")
    if os.path.exists(output_file) and not options.overwrite:
        return
    if os.path.exists(output_file):
        os.remove(output_file)

    # code copied from extract_batch_instance_type
    chunksize = 10_000_000

    start = time.time()
    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1
    lines_read = 0
    # requested_data = pd.read_csv(batch_task_request, names=names, engine="c", header=None)
    timestamp_to_instances = {}
    recording_time_stamp = None
    current_cpu_max = 0.0
    current_cpu_avg = 0.0
    current_mem_max = 0.0
    current_mem_avg = 0.0
    prev_tasks = set()
    not_found = 0
    total = 0
    for j, data in enumerate(pd.read_csv(data_path, names=names, engine="c", chunksize=chunksize, header=None)):
        # the arrays to write (or rather append) to the data path
        timestamps = []
        all_cpu_max = []
        all_cpu_avg = []
        all_mem_max = []
        all_mem_avg = []
        tasks_at_time = []

        total += data[data.columns[0]].count()
        data = data.dropna(subset=["mem_avg", "mem_max", "cpu_avg", "cpu_max"])
        data = data[data['status'] == 'Terminated'].copy()

        start_data = data[["start_time", "instance_name", "task_name", "job_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max"]].copy()
        start_data.columns = ['timestamp', "instance_name", 'task_name', 'job_name', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
        start_data['event_type'] = 'start'
        end_data = data[["end_time", "instance_name", "task_name", "job_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max"]].copy()
        end_data.columns = ['timestamp', "instance_name", 'task_name', 'job_name', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
        end_data['event_type'] = 'end'

        combined_data = pd.concat([start_data, end_data]).sort_values('timestamp')
        # todo: end timestamps at the tail end of the chunk will be dropped, should fix

        data_ndarray = combined_data.to_numpy()
        recording_time_stamp = recording_time_stamp or data_ndarray[0][1]
        # i = 0
        for row in data_ndarray:
            # logger.info(f"{i} of {len(data_ndarray)}")
            # i+=1
            timestamp = row[0]
            instance_name = row[1]
            task_name = row[2]
            job_name = row[3]
            cpu_avg = row[4]
            cpu_max = row[5]
            mem_avg = row[6]
            mem_max = row[7]
            event_type = row[8]
            if event_type == 'start':
                current_cpu_max += cpu_max
                current_cpu_avg += cpu_avg
                current_mem_max += mem_max
                current_mem_avg += mem_avg
                prev_tasks.add((instance_name, task_name, job_name))
            else:
                current_cpu_max -= cpu_max
                current_cpu_avg -= cpu_avg
                current_mem_max -= mem_max
                current_mem_avg -= mem_avg
                if (instance_name, task_name, job_name) in prev_tasks:
                    prev_tasks.remove((instance_name, task_name, job_name))
                else:
                    # logger.warning(f"ending instance {instance_name} task {task_name} with job name {job_name} not found in prev list of tasks {prev_tasks}")
                    not_found += 1
                    pass
            timestamps.append(timestamp)
            all_cpu_max.append(current_cpu_max)
            all_cpu_avg.append(current_cpu_avg)
            all_mem_max.append(current_mem_max)
            all_mem_avg.append(current_mem_avg)
            tasks_at_time.append(prev_tasks)

        logger.info(f"not found percentage: {not_found / total * 100}%")

        parsed_df = pd.DataFrame({"timestamp": timestamps, "cpu_avg": all_cpu_avg, "cpu_max": all_cpu_max, "mem_avg": all_mem_avg, "mem_max": all_mem_max, "tasks": tasks_at_time})
        parsed_df.set_index("timestamp", inplace=True)
        parsed_df.to_csv(output_file, header=False, index=False, mode="a")
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    end = time.time()
    logger.info(f"extracted to {output_file}, took {end - start}")


def plot_hist(data, n_bins, title, output, xlabel, ylabel):
    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    ax1.hist(data, bins=n_bins)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(output)
    plt.clf()
    end = time.time()
    logger.info(f"Plotted to {output} in {end - start} seconds")


def plot_hist_log(data, n_bins, title, output, xlabel, ylabel):
    start = time.time()
    hist, bins = np.histogram(data, bins=n_bins)
    if bins[0] < 0:
        # need to transform data
        # todo: determine why queueing delay has negative data points
        xf = abs(bins[0]) + 1
        bins = bins + xf
        logger.info(f"Transforming bins by {xf} due to negative values in histogram for log transformation")
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    ax1.hist(data, bins=logbins, color="slateblue")
    plt.xscale('log')
    fontsize=12
    ax1.set_title(title, fontsize=fontsize)
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    plt.savefig(output)
    plt.clf()
    end = time.time()
    logger.info(f"Plotted to {output} in {end - start} seconds")


def plot_scatter_log(x, y, title, output, xlabel, ylabel):
    start = time.time()
    log_x = np.logspace(np.min(x), np.log10(np.max(x)) + 1, len(x), base=10)
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    ax1.scatter(log_x, y)
    plt.xscale('log')
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(output)
    plt.clf()
    end = time.time()
    logger.info(f"Plotted to {output} in {end - start} seconds")


def plot_scatter(x, y, title, output, xlabel, ylabel):
    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    ax1.scatter(x, y)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(output)
    plt.clf()
    end = time.time()
    logger.info(f"Plotted to {output} in {end - start} seconds")


def plot_line(x, y, title, output, xlabel, ylabel):
    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    ax1.plot(x, y)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(output)
    plt.clf()
    end = time.time()
    logger.info(f"Plotted to {output} in {end - start} seconds")


def graph_oversubscription(options: argparse.Namespace):
    # batch_task_request = "alibaba-2018.csv"

    # names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
    #                   "status": str, "start_time": np.int32, "end_time": np.int32,
    #                   "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
    #                   "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    # names = list(names_to_types.keys())

    # requested_data = pd.read_csv(batch_task_request, names=names, engine="c", header=None)

    reciprocal_csv = os.path.join("oversubscription_analysis", "running_usage.csv")
    reciprocal_names = ["timestamp", "oversubscription_cpu_avg", "oversubscription_cpu_max", "oversubscription_mem_avg", "oversubscription_mem_max"]

    chunksize = 100_000_000


    num_lines = 2_457_359_682
    max_iter = num_lines // chunksize + 1
    start = time.time()

    data_types = ["cpu_avg", "cpu_max", "mem_avg", "mem_max"]

    num_points = 1_000_000
    cache_file = os.path.join("oversubscription_analysis", f"oversubscription_{num_points}.csv")

    if options.overwrite is True and os.path.exists(cache_file):
        os.remove(cache_file)

    if not os.path.exists(cache_file):
        size = int(num_points / max_iter)
        logger.info(f"looping {max_iter} times and selecting {size} data pts each time")
        for j, data in enumerate(pd.read_csv(reciprocal_csv, names=reciprocal_names, engine="c", chunksize=chunksize, header=None)):
            sampled_data = data.sample(size, replace=False)
            sampled_data.to_csv(cache_file, header=not os.path.exists(cache_file), mode="a")
            logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    # sampled_data_type = pd.read_csv(cache_file)['mem_avg']
    #
    # data_type = 'mem_avg'
    # logger.info(f"Creating oversubscription plot for {data_type} with {len(sampled_data_type)} points...")
    # plot_hist_log(data=sampled_data_type, n_bins=100, title=f"Oversubscription {data_type} (reciprocal)",
    #               output=os.path.join("oversubscription_analysis", f"oversubscription_{data_type}.png"),
    #               xlabel="Log Oversubscription Factor", ylabel="Count")


def calc_metrics_oversubscription(options: argparse.Namespace):
    reciprocal_csv = os.path.join("oversubscription_analysis", "running_usage.csv")
    reciprocal_names = ["timestamp", "oversubscription_cpu_avg", "oversubscription_cpu_max", "oversubscription_mem_avg", "oversubscription_mem_max"]

    chunksize = 100_000_000

    num_lines = 2_457_359_682
    max_iter = num_lines // chunksize + 1
    start = time.time()

    data_types = ["cpu_avg", "cpu_max", "mem_avg", "mem_max"]

    cpu_avg_total, cpu_max_total, mem_avg_total, mem_max_total = 0, 0, 0, 0
    count = 0
    for j, data in enumerate(pd.read_csv(reciprocal_csv, names=reciprocal_names, engine="c", chunksize=chunksize, header=None)):
        count += 1
        cpu_avg_total += data["oversubscription_cpu_avg"].sum()
        cpu_max_total += data["oversubscription_cpu_max"].sum()
        mem_avg_total += data["oversubscription_mem_avg"].sum()
        mem_max_total += data["oversubscription_mem_max"].sum()
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    print(f"Totals are:\n"
          f"cpu_avg_total: {cpu_avg_total}\n"
          f"cpu_max_total: {cpu_max_total}\n"
          f"mem_avg_total: {mem_avg_total}\n"
          f"mem_max_total: {mem_max_total}\n")
    cpu_avg_mean = cpu_avg_total / num_lines
    cpu_max_mean = cpu_max_total / num_lines
    mem_avg_mean = mem_avg_total / num_lines
    mem_max_mean = mem_max_total / num_lines

    print(f"Averages are:\n"
          f"cpu_avg_mean: {cpu_avg_mean}\n"
          f"cpu_max_mean: {cpu_max_mean}\n"
          f"mem_avg_mean: {mem_avg_mean}\n"
          f"mem_max_mean: {mem_max_mean}\n")


def lookup_requested_trace(lst: list, task_name, job_name):
    """
    Do a logarithmic search for task_name and job_name and get the corresponding row
    """
    target1, target2 = job_name, task_name
    target = (target1, target2)
    total_lines = len(lst)
    left, right = 0, total_lines - 1

    while left <= right:
        mid = (left + right) // 2
        row = lst[mid].split(',')

        if row is None:
            break

        key1, key2 = row[0], row[1]

        key = key1, key2

        if key == target:
            return row
        elif key < target:
            left = mid + 1
        else:
            right = mid - 1

        if left > right:
            break
    return None


def reciprocal_bloat_2(options: argparse.Namespace):
    """instead of storing it in a file, run the calculation live"""
    start = time.time()
    data_path = options.trace_file or "../alibaba-trace/batch_instance_sort.csv"
    batch_task_request = "alibaba-2018.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    if not os.path.exists("oversubscription_analysis"):
        os.mkdir("oversubscription_analysis")
    output_file = os.path.join("oversubscription_analysis", "running_usage.csv")
    if os.path.exists(output_file) and not options.overwrite:
        return
    if os.path.exists(output_file):
        os.remove(output_file)

    # # store the file in memory for faster searching
    # with open("alibaba-2018-sorted.csv") as f:
    #     alibaba_trace_lst = f.readlines()[1:]

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
    lines_read = 0
    # requested_data = pd.read_csv(batch_task_request, names=names, engine="c", header=None)
    timestamp_to_instances = {}
    recording_time_stamp = None
    current_cpu_max = 0.0
    current_cpu_avg = 0.0
    current_mem_max = 0.0
    current_mem_avg = 0.0

    # calculating the requested resources sum per loop is too expensive
    # so maintain a running sum
    running_plan_cpu = 0
    running_plan_mem = 0

    prev_tasks = set()
    not_found = 0
    total = 0
    end = time.time()
    logger.info(f"preprocessing done in {end - start} seconds")
    start = time.time()

    # todo: remove temp max timestamp
    min_timestamp = 86400 * 3
    max_timestamp = 86400 * 3
    # early_stop = False
    for j, data in enumerate(pd.read_csv(data_path, names=names, engine="c", chunksize=chunksize, header=None)):
        # the arrays to write (or rather append) to the data path
        timestamps = []
        total_reciprocal_cpu_max = []
        total_reciprocal_cpu_avg = []
        total_reciprocal_mem_max = []
        total_reciprocal_mem_avg = []
        job_names = []
        task_names = []

        total += data[data.columns[0]].count()
        data = data.dropna(subset=["mem_avg", "mem_max", "cpu_avg", "cpu_max"])
        data = data[data['status'] == 'Terminated'].copy()

        start_data = data[["start_time", "instance_name", "task_name", "job_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max"]].copy()
        start_data.columns = ['timestamp', "instance_name", 'task_name', 'job_name', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
        start_data['event_type'] = 'start'
        end_data = data[["end_time", "instance_name", "task_name", "job_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max"]].copy()
        end_data.columns = ['timestamp', "instance_name", 'task_name', 'job_name', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
        end_data['event_type'] = 'end'

        combined_data = pd.concat([start_data, end_data]).sort_values('timestamp')
        # todo: end timestamps at the tail end of the chunk will be dropped, should fix

        data_ndarray = combined_data.to_numpy()
        recording_time_stamp = recording_time_stamp or data_ndarray[0][1]

        INFINITY_SENTINEL = 99999999

        # if early_stop is True:
        #     logger.info('early stopping')
        #     break
        # i = 0
        for row in data_ndarray:
            # logger.info(f"{i} of {len(data_ndarray)}")
            # i+=1
            timestamp = row[0]

            # if int(timestamp) > max_timestamp:
            #     logger.info(f"early stop at {timestamp}")
            #     early_stop = True
            #     break
            instance_name = row[1]
            task_name = row[2]
            job_name = row[3]
            cpu_avg = row[4]
            cpu_max = row[5]
            mem_avg = row[6]
            mem_max = row[7]
            event_type = row[8]
            if event_type == 'start':
                current_cpu_max += cpu_max
                current_cpu_avg += cpu_avg
                current_mem_max += mem_max
                current_mem_avg += mem_avg
                prev_tasks.add((instance_name, task_name, job_name))
            else:
                current_cpu_max -= cpu_max
                current_cpu_avg -= cpu_avg
                current_mem_max -= mem_max
                current_mem_avg -= mem_avg
                if (instance_name, task_name, job_name) in prev_tasks:
                    prev_tasks.remove((instance_name, task_name, job_name))
                else:
                    # logger.warning(f"ending instance {instance_name} task {task_name} with job name {job_name} not found in prev list of tasks {prev_tasks}")
                    not_found += 1
                    pass

            # compute the sum of the requested instances (not what is currently utilized)
            # use a running sum to avoid O(nm) runtime

            plan_resources = alibaba_data[f"{job_name},{task_name}"]

            plan_cpu = plan_resources[0]
            plan_mem = plan_resources[1]
            if event_type == 'start':
                # add to running sums
                running_plan_cpu += plan_cpu
                running_plan_mem += plan_mem
            else:
                running_plan_cpu -= plan_cpu
                running_plan_mem -= plan_mem

            # get back to expected normalized units by dividing by 100
            running_plan_mem /= 100

            reciprocal_cpu_max = running_plan_cpu / current_cpu_max if current_cpu_max != 0 else INFINITY_SENTINEL
            reciprocal_cpu_avg = running_plan_cpu / current_cpu_avg if current_cpu_avg != 0 else INFINITY_SENTINEL
            reciprocal_mem_max = running_plan_mem / current_mem_max if current_mem_max != 0 else INFINITY_SENTINEL
            reciprocal_mem_avg = running_plan_mem / current_mem_avg if current_mem_avg != 0 else INFINITY_SENTINEL

            timestamps.append(timestamp)
            if reciprocal_cpu_max != INFINITY_SENTINEL:
                total_reciprocal_cpu_max.append(reciprocal_cpu_max)
            if reciprocal_cpu_avg != INFINITY_SENTINEL:
                total_reciprocal_cpu_avg.append(reciprocal_cpu_avg)
            if reciprocal_mem_max != INFINITY_SENTINEL:
                total_reciprocal_mem_max.append(reciprocal_mem_max)
            if reciprocal_mem_avg != INFINITY_SENTINEL:
                total_reciprocal_mem_avg.append(reciprocal_mem_avg)
            job_names.append(job_name)
            task_names.append(task_name)

        logger.info(f"not found percentage: {not_found / total * 100}%")

        parsed_df = pd.DataFrame({"timestamp": timestamps, "job_name": job_names, "task_name":  task_names, "oversubscription_cpu_avg": total_reciprocal_cpu_avg, "oversubscription_cpu_max": total_reciprocal_cpu_max,
                                  "oversubscription_mem_avg": total_reciprocal_mem_avg, "oversubscription_mem_max": total_reciprocal_mem_max})
        parsed_df = parsed_df[parsed_df["timestamp"] >= min_timestamp]
        parsed_df = parsed_df[parsed_df["timestamp"] <= max_timestamp]
        parsed_df.to_csv(output_file, header=not os.path.exists(output_file), mode="a", index=False)
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    end = time.time()
    logger.info(f"extracted to {output_file}, took {end - start}")


def sort_requested_trace(options: argparse.Namespace):
    batch_task_request = "alibaba-2018.csv"
    df = pd.read_csv(batch_task_request, header=None, names=['task_name', 'instance_num', 'job_name', 'task_type', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem'])
    df.sort_values(by=["job_name", "task_name"], inplace=True, ascending=True)
    df = df.reindex(columns=["job_name", "task_name", "instance_num", "task_type", "status", "start_time", "end_time", "plan_cpu", "plan_mem"])
    df.to_csv("alibaba-2018-sorted.csv", index=False)


def oversubscription_number(options: argparse.Namespace):
    """
    Calculate oversubscription number of each requested task

    Calculated as the multiple of requested resource over available cluster size
    """
    start = time.time()
    data_path = options.trace_file
    if data_path == "../alibaba-trace/batch_instance.csv":
        data_path = "alibaba-2018.csv"

    requested_df = pd.read_csv(data_path, header=None, names=['task_name', 'instance_num', 'job_name', 'task_type', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem'])

    machine_instance_directory = "machine_instance_analysis"
    remaining_machine_usage = os.path.join(machine_instance_directory, "remaining_machine_usage.csv")
    remaining_df = pd.read_csv(remaining_machine_usage, header=None, names=["timestamp", "remaining_cpu_cluster", "remaining_mem_cluster",
                                                                            "remaining_cpu_machine", "remaining_mem_machine"])

    requested_df["start_time"] = pd.to_numeric(requested_df["start_time"], errors="coerce")
    requested_df["end_time"] = pd.to_numeric(requested_df["end_time"], errors="coerce")

    remaining_df["timestamp"] = pd.to_numeric(remaining_df["timestamp"], errors="coerce")
    print(f"min cpu_cluster: {remaining_df['remaining_cpu_cluster'].min()}")
    print(f"min mem_cluster: {remaining_df['remaining_mem_cluster'].min()}")
    print(f"min cpu_machine: {remaining_df['remaining_cpu_machine'].min()}")
    print(f"min mem_machine: {remaining_df['remaining_mem_machine'].min()}")
    print(f"max cpu_cluster: {remaining_df['remaining_cpu_cluster'].max()}")
    print(f"max mem_cluster: {remaining_df['remaining_mem_cluster'].max()}")
    print(f"max cpu_machine: {remaining_df['remaining_cpu_machine'].max()}")
    print(f"max mem_machine: {remaining_df['remaining_mem_machine'].max()}")
    requested_df_np = requested_df.to_numpy()
    del requested_df

    cpu_cluster_oversubscription_numbers = []
    mem_cluster_oversubscription_numbers = []
    cpu_machine_oversubscription_numbers = []
    mem_machine_oversubscription_numbers = []
    timestamps = []
    infinity_max_value = 9999  # if there are 0 resources avoid divide by 0 and replace infinity with this number
    for row in requested_df_np:
        task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem = row
        # this might be O(n^2)
        remaining_df['diff'] = abs(remaining_df["timestamp"] - start_time)
        closest_row = remaining_df.loc[remaining_df['diff'].idxmin()]
        if closest_row["remaining_cpu_cluster"] == 0:
            cpu_cluster_oversubscription = infinity_max_value
        else:
            cpu_cluster_oversubscription = (plan_cpu / 100 * instance_num) / (4023 * 96 * closest_row["remaining_cpu_cluster"])
        if closest_row["remaining_mem_cluster"] == 0:
            mem_cluster_oversubscription = infinity_max_value
        else:
            mem_cluster_oversubscription = (plan_mem / 100 * instance_num) / (4023 * closest_row["remaining_mem_cluster"])
        if closest_row["remaining_cpu_machine"] == 0:
            cpu_machine_oversubscription = infinity_max_value
        else:
            cpu_machine_oversubscription = (plan_cpu / 100 * instance_num) / (96 * closest_row["remaining_cpu_machine"])
        if closest_row["remaining_mem_machine"] == 0:
            mem_machine_oversubscription = infinity_max_value
        else:
            mem_machine_oversubscription = (plan_mem / 100 * instance_num) / (closest_row["remaining_mem_machine"])

        cpu_cluster_oversubscription_numbers.append(cpu_cluster_oversubscription)
        mem_cluster_oversubscription_numbers.append(mem_cluster_oversubscription)
        cpu_machine_oversubscription_numbers.append(cpu_machine_oversubscription)
        mem_machine_oversubscription_numbers.append(mem_machine_oversubscription)
        timestamps.append(start_time)

    df = pd.DataFrame({"timestamp": timestamps, "cpu_cluster_oversubscription": cpu_cluster_oversubscription_numbers,
                       "mem_cluster_oversubscription": mem_cluster_oversubscription_numbers,
                       "cpu_machine_oversubscription": cpu_machine_oversubscription_numbers,
                       "mem_machine_oversubscription": mem_machine_oversubscription_numbers},
                      columns=["timestamp", "cpu_cluster_oversubscription", "mem_cluster_oversubscription", "cpu_machine_oversubscription", "mem_machine_oversubscription"])

    output_file = os.path.join(machine_instance_directory, "oversubscription_numbers.csv")
    df.to_csv(output_file, index=False)
    end = time.time()
    print(f"wrote to {output_file} in {end - start} seconds")


def plot_oversubscription_number_over_timestamps(options: argparse.Namespace):
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    start = time.time()
    machine_instance_directory = "machine_instance_analysis"
    oversubscription_file = os.path.join(machine_instance_directory, "oversubscription_numbers.csv")

    names = ["timestamp", "cpu_cluster_oversubscription", "mem_cluster_oversubscription", "cpu_machine_oversubscription", "mem_machine_oversubscription"]
    df = pd.read_csv(oversubscription_file)
    for name in names:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    cluster_plot_filename = os.path.join(machine_instance_directory, "oversubscription_cluster_graph.png")
    create_plot(list(df["timestamp"]), list(df["cpu_cluster_oversubscription"]), list(df["mem_cluster_oversubscription"]), input_file_name=os.path.basename(oversubscription_file),
                output_directory=machine_instance_directory, write_config=False, cores_label="CPU cluster oversubscription factor",
                mem_label="Mem cluster oversubscription factor", output=cluster_plot_filename)

    machine_plot_filename = os.path.join(machine_instance_directory, "oversubscription_machine_graph.png")
    create_plot(list(df["timestamp"]), list(df["cpu_machine_oversubscription"]), list(df["mem_machine_oversubscription"]), input_file_name=os.path.basename(oversubscription_file),
                output_directory=machine_instance_directory, write_config=False, cores_label="CPU machine oversubscription factor",
                mem_label="Mem machine oversubscription factor", output=machine_plot_filename)

    end = time.time()
    print(f"Outputted to {cluster_plot_filename} and {machine_plot_filename} in {end - start} seconds")


def plot_oversubscription_number(options: argparse.Namespace):
    """
    Plot distribution of oversubscription numbers
    """
    machine_instance_directory = "machine_instance_analysis"
    oversubscription_file = os.path.join(machine_instance_directory, "oversubscription_numbers.csv")

    names = ["timestamp", "cpu_cluster_oversubscription", "mem_cluster_oversubscription", "cpu_machine_oversubscription", "mem_machine_oversubscription"]
    df = pd.read_csv(oversubscription_file)

    for name in names:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    print(f"min cpu_cluster: {df['cpu_cluster_oversubscription'].min()}")
    print(f"min mem_cluster: {df['mem_cluster_oversubscription'].min()}")
    print(f"min cpu_machine: {df['cpu_machine_oversubscription'].min()}")
    print(f"min mem_machine: {df['mem_machine_oversubscription'].min()}")
    print(f"max cpu_cluster: {df['cpu_cluster_oversubscription'].max()}")
    print(f"max mem_cluster: {df['mem_cluster_oversubscription'].max()}")
    print(f"max cpu_machine: {df['cpu_machine_oversubscription'].max()}")
    print(f"max mem_machine: {df['mem_machine_oversubscription'].max()}")

    n_bins = 100

    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    name = "cpu_cluster_oversubscription"
    ax1.hist(df[f"{name}"], bins=n_bins)
    ax1.set_title(f"{name} factor")
    ax1.set_xlabel("oversubscription factor")
    ax1.set_ylabel("count")
    plot_filename = os.path.join(machine_instance_directory, f"{name}.png")
    plt.savefig(plot_filename)
    plt.clf()
    end = time.time()
    print(f"Plotted to {plot_filename} in {end - start} seconds")

    start = time.time()
    fig, ax1 = plt.subplots(tight_layout=True)
    name = "mem_cluster_oversubscription"
    ax1.hist(df[f"{name}"], bins=n_bins)
    ax1.set_title(f"{name} factor")
    ax1.set_xlabel("oversubscription factor")
    ax1.set_ylabel("count")
    plot_filename = os.path.join(machine_instance_directory, f"{name}.png")
    plt.savefig(plot_filename)
    end = time.time()
    print(f"Plotted to {plot_filename} in {end - start} seconds")
    plt.clf()

    start = time.time()
    fig, ax1 = plt.subplots(tight_layout=True)
    name = "cpu_machine_oversubscription"
    ax1.hist(df[f"{name}"], bins=n_bins)
    ax1.set_title(f"{name} factor")
    ax1.set_xlabel("oversubscription factor")
    ax1.set_ylabel("count")
    plot_filename = os.path.join(machine_instance_directory, f"{name}.png")
    plt.savefig(plot_filename)
    end = time.time()
    print(f"Plotted to {plot_filename} in {end - start} seconds")
    plt.clf()

    start = time.time()
    fig, ax1 = plt.subplots(tight_layout=True)
    name = "mem_machine_oversubscription"
    ax1.hist(df[f"{name}"], bins=n_bins)
    ax1.set_title(f"{name} factor")
    ax1.set_xlabel("oversubscription factor")
    ax1.set_ylabel("count")
    plot_filename = os.path.join(machine_instance_directory, f"{name}.png")
    plt.savefig(plot_filename)
    end = time.time()
    print(f"Plotted to {plot_filename} in {end - start} seconds")
    plt.clf()


def real_queueing_delay(options: argparse.Namespace):
    batch_file = options.trace_file or "../alibaba-trace/batch_instance_sort.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    if not os.path.exists("batch_instance_analysis"):
        os.mkdir("batch_instance_analysis")
    output_file = os.path.join("batch_instance_analysis", "queueing_delays.csv" if options.iteration is None else f"queueing_delays_{options.iteration}.csv")
    if os.path.exists(output_file) and not options.overwrite:
        return
    if os.path.exists(output_file):
        os.remove(output_file)

    alibaba_trace = pd.read_csv("alibaba-2018-sorted.csv")
    alibaba_trace = alibaba_trace[["job_name", "task_name", "start_time", "end_time"]]
    alibaba_dict = {f"{row[0]},{row[1]}": [int(row[2]), int(row[3])] for row in alibaba_trace.to_numpy()}
    del alibaba_trace
    start = time.time()

    # code copied from extract_batch_instance_type
    chunksize = 100_000

    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1

    # iter = 0 if options.iteration is None else int(options.iteration)
    # num_chunks = 4
    # iter_chunk = math.ceil(max_iter / num_chunks)
    # start_iter = iter_chunk * iter
    # stop_iter = iter_chunk * (iter + 1) if options.iteration is not None else max_iter + 1

    total = 0
    not_found = 0
    recording_time_stamp = None
    for j, data in enumerate(pd.read_csv(batch_file, names=names, engine="c", chunksize=chunksize, header=None)):
        # if j < start_iter:
        #     logger.info(f"skipping iter {j} to get to {start_iter}")
        #     continue
        # if j >= stop_iter:
        #     logger.info(f"early stop iter {j} because stop at {stop_iter}")
        #     break
        # the arrays to write (or rather append) to the data path
        timestamps = []
        q_delays = []
        job_names = []
        task_names = []
        req_times = []

        total += data[data.columns[0]].count()
        data = data.dropna(subset=["mem_avg", "mem_max", "cpu_avg", "cpu_max"])
        data = data[data['status'] == 'Terminated'].copy()

        start_data = data[["start_time", "instance_name", "task_name", "job_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max", "start_time", "end_time"]].copy()
        start_data.columns = ['timestamp', "instance_name", 'task_name', 'job_name', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max', "start_time", "end_time"]
        start_data['event_type'] = 'start'
        end_data = data[["end_time", "instance_name", "task_name", "job_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max", "start_time", "end_time"]].copy()
        end_data.columns = ['timestamp', "instance_name", 'task_name', 'job_name', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max', "start_time", "end_time"]
        end_data['event_type'] = 'end'

        combined_data = pd.concat([start_data, end_data]).sort_values('timestamp')
        # todo: end timestamps at the tail end of the chunk will be dropped, should fix

        data_ndarray = combined_data.to_numpy()
        recording_time_stamp = recording_time_stamp or data_ndarray[0][1]

        INFINITY_SENTINEL = 9999
        # i = 0
        for row in data_ndarray:
            # logger.info(f"{i} of {len(data_ndarray)}")
            # i+=1
            timestamp = row[0]
            instance_name = row[1]
            task_name = row[2]
            job_name = row[3]
            cpu_avg = row[4]
            cpu_max = row[5]
            mem_avg = row[6]
            mem_max = row[7]
            event_type = row[10]
            start_time = int(row[8])
            end_time = row[9]
            if event_type == 'start':
                requested_time = alibaba_dict[f"{job_name},{task_name}"][0]
                # real start - request, this is how long it took to schedule the job since the request
                # where request is the time the task starts and the real start is the time an instance starts
                # therefore, since there can be multiple instnacees, record them all
                diff = int(start_time) - requested_time
                # logger.info(f"{job_name},{task_name} has req time {requested_time} with real start {start_time}")
                req_times.append(requested_time)
                q_delays.append(diff)
                timestamps.append(timestamp)
                job_names.append(job_name)
                task_names.append(task_name)

        parsed_df = pd.DataFrame({"timestamp": timestamps, "queueing_delay": q_delays, "job_name": job_names, "task_name": task_names, "requested_time": req_times})
        parsed_df.set_index("timestamp", inplace=True)
        parsed_df.to_csv(output_file, header=not os.path.exists(output_file), mode="a")
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    end = time.time()
    logger.info(f"grabbed qdelay to {output_file}, took {end - start}")


def real_queueing_delay_dag(options: argparse.Namespace):
    dag_sampling_import = import_module("dag")
    parse_task_name_with_prefix = getattr(dag_sampling_import, "parse_task_name_with_prefix")

    batch_file = "../alibaba-trace/batch_instance_sort.csv"

    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())

    if not os.path.exists("batch_instance_analysis"):
        os.mkdir("batch_instance_analysis")
    output_file = os.path.join("batch_instance_analysis", "queueing_delays_dag.csv" if options.iteration is None else f"queueing_delays_{options.iteration}.csv")
    if os.path.exists(output_file) and not options.overwrite:
        return
    if os.path.exists(output_file):
        os.remove(output_file)

    alibaba_trace = pd.read_csv("alibaba-2018-sorted.csv")
    alibaba_trace.dropna(inplace=True)
    alibaba_trace = alibaba_trace[["job_name", "task_name", "start_time", "end_time"]]
    # alibaba_trace = alibaba_trace[["start_time", "end_time", "plan_cpu", "plan_mem", "instance_num", "status", "job_name", "task_name"]]
    # alibaba_dict = {f"{row[6]},{row[7]}": [int(row[1]), int(row[2])] for row in alibaba_trace.to_numpy()}

    # alibaba_dict = {f"{row[0]},{row[1]}": [int(row[2]), int(row[3])] for row in alibaba_trace.to_numpy()}

    # easy but inefficient solution
    # i need to map job to tasks when looking up dependent tasks as I dont know what the dependent task name is (as in, i could look for task 13 but the name is 13_14)
    job_to_tasks = dict()
    alibaba_dict = dict()
    for row in alibaba_trace.to_numpy():
        job_name, task_name, start_time, end_time = row
        alibaba_dict[f"{row[0]},{row[1]}"] = [int(row[2]), int(row[3])]
        job_to_tasks.setdefault(job_name, dict())
        # subdict that maps first task number to the actual name, as in if task name is 13_14 then map 13 to 13_14
        if task_name[0] == 't':
            # skip the individual tasks
            continue
        base_task = task_name[1:].split("_")[0]
        job_to_tasks[job_name][base_task] = task_name

    # see https://stackoverflow.com/a/7971655
    from itertools import islice
    def take(n, iterable):
        """Return the first n items of the iterable as a list."""
        return list(islice(iterable, n))

    logger.info(f"{take(10, job_to_tasks.items())}")
    # todo: the current get_dags function throws away the job if a dependency does not exist at schedule time.
    # it may be possible that the task dependency will exist later, but there are still holes in the data
    # so refactor get_dags or make a new function to return a structure without this check
    # job_to_dag = get_dags(alibaba_trace)
    # parse through the alibaba metadata dict of start and end times to make the start times dag aware
    start = time.time()
    changed = 0
    not_found = 0
    total = 0
    oddities = 0

    for k, v in alibaba_dict.items():
        job_name, task_name = k.split(',')

        # for each job, task combo, make sure the task start time is the very beginning
        # the task name should indicate the dependencies of the task per job
        task_prefix, base_task, dependent_tasks = parse_task_name_with_prefix(task_name)
        if task_prefix is None or len(dependent_tasks) == 0:
            # this is an independent task so skip
            continue
        latest = 0
        # print(f"Job name {job_name} with task name {task_name}")
        for task in dependent_tasks:
            # for all dependent tasks choose the latest start time

            # according to the schema the prefix has nothing to do with the dependency
            # but it seems like the prefix can be any character so ensure we look up multiple
            # from testing: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'J': 1380155, 'K': 0, 'L': 663,
            # 'M': 5981963, 'N': 0, 'O': 0, 'P': 0, 'Q': 0, 'R': 4844922, 'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0, 'Y': 0, 'Z': 0}
            # possible_prefixes = ["J", "L", "M", "R"]
            #
            # dont need the above with a new mapping
            task_name = job_to_tasks.get(job_name).get(task)
            dependent_task = alibaba_dict.get(f"{job_name},{task_name}")
            total += 1
            if dependent_task is None or task_name is None:
                not_found += 1
                logger.info(f"Not found dependent task {job_name} {task} with parent task {task_name}")
                continue
            end_time = dependent_task[1]
            if end_time > latest:
                latest = end_time

        if latest > v[0]:
            # the task was requested before scheduleable so change the start time
            if latest > v[1]:
                # task scheduleable is after written end time
                # logger.info(f"Job {k} with start/end times {v} with new latest {latest} detected")
                oddities += 1
            alibaba_dict[k][0] = latest
            changed += 1

    end = time.time()
    logger.info(f"Unable to lookup {not_found} times out of a total lookup {total} times")
    logger.info(f"Changed {changed} total job/task pairs out of {len(alibaba_dict)} for {changed / len(alibaba_dict) * 100}%... took {end - start} seconds")
    logger.info(f"There exists {oddities} times where the new start time is after the reported finish time of a task for a total of {oddities / total * 100}%")
    start = time.time()

    # code copied from extract_batch_instance_type
    chunksize = 10_000_000

    num_lines = 1_351_255_775
    max_iter = num_lines // chunksize + 1

    # iter = 0 if options.iteration is None else int(options.iteration)
    # num_chunks = 4
    # iter_chunk = math.ceil(max_iter / num_chunks)
    # start_iter = iter_chunk * iter
    # stop_iter = iter_chunk * (iter + 1) if options.iteration is not None else max_iter + 1

    total = 0
    not_found = 0
    recording_time_stamp = None
    for j, data in enumerate(pd.read_csv(batch_file, names=names, engine="c", chunksize=chunksize, header=None)):
        # if j < start_iter:
        #     logger.info(f"skipping iter {j} to get to {start_iter}")
        #     continue
        # if j >= stop_iter:
        #     logger.info(f"early stop iter {j} because stop at {stop_iter}")
        #     break
        # the arrays to write (or rather append) to the data path
        timestamps = []
        q_delays = []
        job_names = []
        task_names = []
        completion_times = []
        job_length = []
        instance_names = []
        cpu_avgs = []
        cpu_maxes = []
        mem_avgs = []
        mem_maxes = []

        total += data[data.columns[0]].count()
        data = data.dropna(subset=["mem_avg", "mem_max", "cpu_avg", "cpu_max"])
        data = data[data['status'] == 'Terminated'].copy()

        parsed_data = data[["start_time", "instance_name", "task_name", "job_name", "cpu_avg", "cpu_max", "mem_avg", "mem_max", "start_time", "end_time"]]
        parsed_data.columns = ['timestamp', "instance_name", 'task_name', 'job_name', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max', "start_time", "end_time"]

        parsed_data = parsed_data.sort_values('timestamp')
        # todo: end timestamps at the tail end of the chunk will be dropped, should fix

        data_ndarray = parsed_data.to_numpy()
        recording_time_stamp = recording_time_stamp or data_ndarray[0][1]

        INFINITY_SENTINEL = 9999
        # i = 0
        for row in data_ndarray:
            # logger.info(f"{i} of {len(data_ndarray)}")
            # i+=1
            timestamp = row[0]
            instance_name = row[1]
            task_name = row[2]
            job_name = row[3]
            cpu_avg = row[4]
            cpu_max = row[5]
            mem_avg = row[6]
            mem_max = row[7]
            start_time = int(row[8])
            end_time = int(row[9])
            # note: this currently means earliest scheduleable time
            scheduleable_time = alibaba_dict[f"{job_name},{task_name}"][0]
            # real start - request, this is how long it took to schedule the job since the request
            # where request is the time the task starts and the real start is the time an instance starts
            # therefore, since there can be multiple instnacees, record them all
            diff = int(start_time) - scheduleable_time
            # logger.info(f"{job_name},{task_name} has req time {requested_time} with real start {start_time}")
            q_delays.append(diff)
            timestamps.append(timestamp)
            job_names.append(job_name)
            task_names.append(task_name)
            completion_times.append(end_time - scheduleable_time)
            job_length.append(end_time - start_time)
            instance_names.append(instance_name)
            cpu_avgs.append(cpu_avg)
            cpu_maxes.append(cpu_max)
            mem_avgs.append(mem_avg)
            mem_maxes.append(mem_max)

        parsed_df = pd.DataFrame({"timestamp": timestamps, "queueing_delay": q_delays, "completion_time": completion_times, "job_name": job_names, "task_name": task_names,
                                  "job_length": job_length, "cpu_avg": cpu_avgs, "cpu_max": cpu_maxes, "mem_avg": mem_avgs, "mem_max": mem_maxes})
        parsed_df.set_index("timestamp", inplace=True)
        parsed_df.to_csv(output_file, header=not os.path.exists(output_file), mode="a")
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    end = time.time()
    logger.info(f"grabbed qdelay to {output_file}, took {end - start}")


def calc_metrics_queueing_delay(options: argparse.Namespace):
    queueing_delay_csv = os.path.join("batch_instance_analysis", "queueing_delays_dag.csv")

    chunksize = 100_000_000

    num_lines = 1_228_679_841
    max_iter = num_lines // chunksize + 1
    start = time.time()

    qdelay_total = 0
    count = 0
    for j, data in enumerate(pd.read_csv(queueing_delay_csv, engine="c", chunksize=chunksize, header=None)):
        count += 1
        qdelay_total += data["queueing_delay"].sum()
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    mean_value = qdelay_total / num_lines

    print(f"Averages are:\n"
          f"queueing_delay average: {mean_value}\n")


def dump_queueing_delay(options: argparse.Namespace):
    df = None
    for i in range(4):
        f = f"batch_instance_analysis/queueing_delays_{i}.csv"
        if df is None:
            df = pd.read_csv(f)
        else:
            df = df.merge(pd.read_csv(f))

    queueing_delay = "batch_instance_analysis/queueing_delays_merged.csv"
    df.to_csv(queueing_delay)


def graph_queueing_delay(options: argparse.Namespace):
    # queueing_delay = "batch_instance_analysis/queueing_delays_merged.csv"
    num_points = options.num_points or 1_000_000
    dag = options.dag or True
    if dag:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_dag_{num_points}.csv")
        line_plot = os.path.join("batch_instance_analysis", "queueing_delays_dag_line_graph.png")
        completion_cdf = os.path.join("batch_instance_analysis", "queueing_delays_completion_times_dag_cdf.png")
        completion_graph = os.path.join("batch_instance_analysis", "queueing_delays_completion_times_dag_graph.png")
        distribution_plot = os.path.join("batch_instance_analysis", "queueing_delays_dag_graph_hist.png")
        scatter_plot = os.path.join("batch_instance_analysis", "queueing_delays_completion_times_dag.png")
        scatter_plot_log = os.path.join("batch_instance_analysis", "queueing_delays_completion_times_dag_log.png")
        queueing_delay = "batch_instance_analysis/queueing_delays_dag.csv"
    else:
        cache_file = os.path.join("batch_instance_analysis", f"qdelay_pts_{num_points}.csv")
        line_plot = os.path.join("batch_instance_analysis", "queueing_delays_line_graph.png")
        distribution_plot = os.path.join("batch_instance_analysis", "queueing_delays_graph_hist.png")
        scatter_plot = os.path.join("batch_instance_analysis", "queueing_delays_completion_times.png")
        scatter_plot_log = os.path.join("batch_instance_analysis", "queueing_delays_completion_times_log.png")
        queueing_delay = "batch_instance_analysis/queueing_delays.csv"
        completion_cdf = os.path.join("batch_instance_analysis", "queueing_delays_completion_times_cdf.png")
        completion_graph = os.path.join("batch_instance_analysis", "queueing_delays_completion_times_graph.png")
    num_lines = 1_228_679_841
    n_bins = 100

    start = time.time()
    if os.path.exists(cache_file) and not options.overwrite:
        df = pd.read_csv(cache_file)
        timestamps = df["timestamp"]
        qdelay_pts = df["queueing_delay"]
        completion_times = df["completion_time"]
        del df
    else:
        if os.path.exists(cache_file):
            os.remove(cache_file)
        qdelay_pts = []
        timestamps = []
        completion_times = []
        chunksize = 100_000_000
        max_iter = num_lines // chunksize + 1
        size = int(num_points / max_iter)
        logger.info(f"looping {max_iter} times and selecting {size} data pts each time")
        for j, data in enumerate(
                pd.read_csv(queueing_delay, engine="c", chunksize=chunksize, header=None)):
            sampled_data = data.sample(size, replace=False)
            qdelay_pts.extend(sampled_data["queueing_delay"])
            timestamps.extend(sampled_data["timestamp"])
            completion_times.extend(sampled_data["completion_time"])
            logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
            sampled_data.to_csv(cache_file, header=False, mode="a")

    plot_hist_log(data=qdelay_pts, n_bins=n_bins, title="Queuing delay hist", output=distribution_plot,
                  xlabel="Log qdelay", ylabel="Count")

    def remove_negative(l):
        if l[1] < 0:
            return False
        return True

    timestamps, qdelay_pts = (list(t) for t in zip(*sorted(filter(remove_negative, zip(timestamps, qdelay_pts)))))
    completion_times, qdelay_pts = (list(t) for t in zip(*sorted(filter(remove_negative, zip(completion_times, qdelay_pts)))))

    plot_scatter(completion_times, qdelay_pts, title="Queuing delay over completion time", output=scatter_plot,
                 xlabel="completion time", ylabel="qdelay")
    plot_scatter_log(completion_times, qdelay_pts, title="Queuing delay over completion time (log)", output=scatter_plot_log,
                     xlabel="completion time (log)", ylabel="qdelay")

    plot_scatter(timestamps, qdelay_pts, title="Queueing delay", output=line_plot, xlabel="timestamp", ylabel="qdelay")
    plot_scatter(completion_times, qdelay_pts, title="Queueing delay over completion time", output=completion_graph, xlabel="completion times", ylabel="qdelay")
    #x, y = calculate_cdf(qdelay_pts)
    #plot_line(x, y, title="Queueing delay CDF", output=completion_cdf, ylabel="%", xlabel="Queueing delay")
    from critical_path_qdelay import plot_cdf_log_x
    plot_cdf_log_x(qdelay_pts, title='CDF of Queueing Delay', output=completion_cdf, ylabel='%', xlabel="Queueing delay in seconds (log)")

    # qdelay over completion time similar to critical path
    ratios_queueing_delay_over_completion = []
    new_timestamps = []
    for completion_time, qdelay, timestamp in zip(completion_times, qdelay_pts, timestamps):
        # each zip entry should represent an individual timestamp
        if completion_time != 0:
            ratio_queueing_delay_over_completion_time_in_critical_path = qdelay / completion_time
            ratios_queueing_delay_over_completion.append(ratio_queueing_delay_over_completion_time_in_critical_path)
            new_timestamps.append(timestamp)

    plot_cdf_log_x(ratios_queueing_delay_over_completion, 'Ratio of qdelay over completion time', 'CDF',
                   'CDF of qdelay over completion', os.path.join("batch_instance_analysis", "ratio_queueing_delay_over_completion_dag.png"))


def negative_queueing_delay(options: argparse.Namespace):
    # for some reason there are negative queueing delay values so get the job name and task names of them
    dag = options.dag or True
    if dag:
        queueing_delay = "batch_instance_analysis/queueing_delays_dag.csv"
        out_file = os.path.join("batch_instance_analysis", "queueing_delays_dag_negative.csv")
    else:
        queueing_delay = "batch_instance_analysis/queueing_delays.csv"
        out_file = os.path.join("batch_instance_analysis", "queueing_delays_negative.csv")
    start = time.time()
    num_lines = 1_228_679_841
    chunksize = 100_000_000
    max_iter = num_lines // chunksize + 1
    if os.path.exists(out_file) and not options.overwrite:
        logger.info(f"set overwrite to overwrite {out_file}")
    if os.path.exists(out_file):
        os.remove(out_file)
    for j, data in enumerate(
            pd.read_csv(queueing_delay, engine="c", chunksize=chunksize, header=None)):
        # less than 0 for now, but there probably are 0 values
        rows = data[data["queueing_delay"] < 0]
        rows.set_index("timestamp", inplace=True)
        rows.to_csv(out_file, header=False, mode="a")
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    end = time.time()
    logger.info(f"Extracted to {out_file}, took {end - start}")


def test_function(options: argparse.Namespace):
    alibaba_trace = pd.read_csv("batch_instance_analysis/queueing_delays_dag_negative.csv", names=["timestamp", "qdelay", "job_name", "task_name"])

    alibaba_trace = alibaba_trace[["qdelay"]]
    min_q, max_q = float('inf'), float('-inf')
    for row in alibaba_trace.to_numpy():
        num = int(row[0])
        if num < min_q:
            min_q = num
        if num > max_q:
            max_q = num
    print(f"Min negative is {min_q} and max is {max_q}")


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", "-t", help="Where the machine_usage trace is", default="../alibaba-trace/batch_instance.csv")
    parser.add_argument("--operation", "-o", help="The operation to run on the trace")
    parser.add_argument("--output", help="output directory", default="batch_instance_analysis")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--norm", default=False, action="store_true", help="Whether to normalize data when graphing")
    parser.add_argument("--infile", default=None)
    parser.add_argument("--instance-to-task-estimation", "-e", help="What algorithm to use when converting instance level data to task level data. Ex: max, avg",
                        default="avg")
    parser.add_argument("--lookup-job")
    parser.add_argument("--lookup-task")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--num-points", type=int, default=1_000_000)
    parser.add_argument("--dag", action="store_true", default=False)
    options = parser.parse_args(args)

    operation = options.operation

    if operation == "avg":
        extract_batch_instance_type(options, type="avg")
        plot_batch_instance_type(options, type="avg")
    elif operation == "avg_plot":
        plot_batch_instance_type(options, type="avg")
    elif operation == "max":
        extract_batch_instance_type(options, type="max")
        plot_batch_instance_type(options, type="max")
    elif operation == "max_plot":
        plot_batch_instance_type(options, type="max")
    elif operation == "get_unique_status":
        get_unique_status(options)
    elif operation == "batch_instance_to_task":
        get_batch_instance_to_task_usage(options)
    elif operation == "compare_batch_instance_to_task":
        compare_batch_instance_to_task(options)
    elif operation == "plot_batch_instance_to_task_pdf":
        plot_batch_instance_to_task_pdf(options)
    elif operation == "plot_batch_instance_to_task_cdf":
        plot_batch_instance_to_task_cdf(options)
    elif operation == "get_instances_of_job_task":
        get_instances_of_job_task(options)
    elif operation == "instances_of_job_task_sort":
        instances_of_job_task_sort(options)
    elif operation == "extract_machine_usage":
        extract_machine_usage(options)
    elif operation == "plot_remaining_machine_usage":
        plot_remaining_machine_usage(options)
    elif operation == "oversubscription_number":
        oversubscription_number(options)
    elif operation == "plot_oversubscription_number":
        plot_oversubscription_number(options)
    elif operation == "plot_oversubscription_number_over_timestamps":
        plot_oversubscription_number_over_timestamps(options)
    elif operation == "reciprocal_bloat":
        # reciprocal_bloat(options)
        reciprocal_bloat_2(options)
    elif operation == "test_start_end_timestamp":
        test_start_end_timestamp(options)
    elif operation == "sort_requested_trace":
        sort_requested_trace(options)
    elif operation == "lookup_requested_trace":
        with open("alibaba-2018-sorted.csv") as f:
            lines = f.readlines()[1:]
        print(lookup_requested_trace(lines, "M1", "j_1"))
    elif operation == "real_queueing_delay":
        real_queueing_delay(options)
    elif operation == "real_queueing_delay_dag":
        real_queueing_delay_dag(options)
    elif operation == "graph_queueing_delay":
        graph_queueing_delay(options)
    elif operation == "dump_queueing_delay":
        dump_queueing_delay(options)
    elif operation == "graph_oversubscription":
        graph_oversubscription(options)
    elif operation == "calc_metrics_oversubscription":
        calc_metrics_oversubscription(options)
    elif operation == "calc_metrics_queueing_delay":
        calc_metrics_queueing_delay(options)
    elif operation == "negative_queueing_delay":
        negative_queueing_delay(options)
    elif operation == "test_function":
        test_function(options)


if __name__ == "__main__":
    main()

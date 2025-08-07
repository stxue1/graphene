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

def real_queueing_delay_dag(options: argparse.Namespace):
    """

    """
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
    output_file = os.path.join("batch_instance_analysis", "queueing_delays_dag.csv")

    if os.path.exists(output_file) and not options.overwrite:
        return
    if os.path.exists(output_file):
        os.remove(output_file)

    alibaba_trace = pd.read_csv("alibaba-2018-sorted.csv")
    alibaba_trace.dropna(inplace=True)
    # the following may be a slice of the data, so make sure to drop all na
    requested_resources = pd.read_csv("cluster_analysis/machine_usage_subscribed.csv").sort_values(by="time_stamp")
    alibaba_trace = alibaba_trace[["job_name", "task_name", "start_time", "end_time", "plan_cpu", "plan_mem", "instance_num"]].sort_values(by="start_time")
    lowest_machine_utilization = pd.read_csv("cluster_analysis/machine_usage_lowest_individual.csv").sort_values(by="time_stamp")
    cluster_utilization = pd.read_csv("cluster_analysis/machine_usage_all.csv").sort_values(by="time_stamp")
    last_timestamp = requested_resources["time_stamp"].max()
    first_timestamp = requested_resources["time_stamp"].min()
    # clip the other traces to the range of the requested resources as that is currently the main limiting factor
    # todo: run on fuller portion of req res
    # this currently has a side effect where jobs near the end and start of the cutoff may not have their start_times be properly set to the dag aware earliest scheduleable time
    alibaba_trace = alibaba_trace[(alibaba_trace["start_time"] > first_timestamp) & (alibaba_trace["start_time"] < last_timestamp)]
    lowest_machine_utilization = lowest_machine_utilization[(lowest_machine_utilization["time_stamp"] > first_timestamp) & (lowest_machine_utilization["time_stamp"] < last_timestamp)]
    # merge the two DF by timestamp--start_time distance. it should be "backward", as in choose the timestamp right before the start_time
    # this will map the cluster subscription to the requested time of the task
    # todo: could I also calc oversubscription ratio by modifying cluster_analyze? aka for each instance, ratio req/used
    # I may need to merge req with batch_instance (or make lookup table to get req resources) in cluster_analyze.py
    # I would also need to input it into a different dataframe and read that here
    req_with_util = pd.merge_asof(lowest_machine_utilization, requested_resources, on="time_stamp", direction="backward")
    req_with_machine_and_cluster_util = pd.merge_asof(cluster_utilization, req_with_util, on="time_stamp", direction="backward")
    alibaba_trace_with_req_and_util = pd.merge_asof(alibaba_trace, req_with_machine_and_cluster_util, left_on="start_time", right_on="time_stamp", direction="backward")
    alibaba_trace_with_req_and_util.dropna(inplace=True)
    alibaba_trace_with_req_and_util = alibaba_trace_with_req_and_util[["job_name", "task_name", "start_time", "end_time", "cpu_subscribed", "mem_subscribed", "cpu_util_percent_min",
                                                                       "mem_util_percent_min", "cpu_util_percent_max", "mem_util_percent_max", "cpu_util_percent",
                                                                       "mem_util_percent", "plan_cpu", "plan_mem", "instance_num"]].sort_values("start_time")
    logger.warning(f"NOTE TO SELF, currently dividing mem subscribed by 100 due to cluster_analyze.py multiplying by 100 for efficiency reasons. "
                   f"Remove this operation if ever changed.")
    # this will return the values back to the raw values
    alibaba_trace_with_req_and_util["mem_subscribed"] /= 100
    del alibaba_trace
    del requested_resources

    # easy but inefficient solution
    # i need to map job to tasks when looking up dependent tasks as I dont know what the dependent task name is (as in, i could look for task 13 but the name is 13_14)
    job_to_tasks = dict()
    alibaba_dict = dict()
    for row in alibaba_trace_with_req_and_util.to_numpy():
        (job_name, task_name, start_time, end_time, cpu_subscribed, mem_subscribed, cpu_util_percent_min, mem_util_percent_min, cpu_util_percent_max, mem_util_percent_max,
         cpu_util_percent_cluster, mem_util_percent_cluster, plan_cpu, plan_mem, instance_num) = row
        alibaba_dict[f"{job_name},{task_name}"] = [int(start_time), int(end_time), cpu_subscribed, mem_subscribed, cpu_util_percent_min, mem_util_percent_min,
                                                   cpu_util_percent_max, mem_util_percent_max, cpu_util_percent_cluster, mem_util_percent_cluster, plan_cpu, plan_mem, instance_num]
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
                # logger.info(f"Not found dependent task {job_name} {task} with parent task {task_name}")
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
        at_req_cpu_subscribed_all = []
        at_req_mem_subscribed_all = []
        at_req_cpu_util_min = []
        at_req_mem_util_min = []
        at_req_cpu_util_max = []
        at_req_mem_util_max = []
        at_req_cpu_cluster_util_all = []
        at_req_mem_cluster_util_all = []
        plan_cpus = []
        plan_mems = []
        instance_nums = []

        total += data[data.columns[0]].count()
        data = data.dropna(subset=["mem_avg", "mem_max", "cpu_avg", "cpu_max"])
        # todo: dont drop with full subscribed trace
        data = data[(data["start_time"] > first_timestamp) & (data["start_time"] < last_timestamp)]
        if data.shape[0] == 0:
            continue

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
            at_req_info = alibaba_dict.get(f"{job_name},{task_name}")
            if at_req_info is None:
                not_found += 1
                continue
            scheduleable_time = at_req_info[0]
            at_req_cpu_subscribed = at_req_info[2]
            at_req_mem_subscribed = at_req_info[3]
            at_req_lowest_machine_cpu_util_percent = at_req_info[4]
            at_req_lowest_machine_mem_util_percent = at_req_info[5]
            # this is more out of curiosity
            at_req_highest_machine_cpu_util_percent = at_req_info[6]
            at_req_highest_machine_mem_util_percent = at_req_info[7]
            at_req_cluster_cpu_util_percent = at_req_info[8]
            at_req_cluster_mem_util_percent = at_req_info[9]
            plan_cpu = at_req_info[10]
            plan_mem = at_req_info[11]
            instance_num = at_req_info[12]
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
            at_req_cpu_subscribed_all.append(at_req_cpu_subscribed)
            at_req_mem_subscribed_all.append(at_req_mem_subscribed)
            at_req_cpu_util_min.append(at_req_lowest_machine_cpu_util_percent)
            at_req_mem_util_min.append(at_req_lowest_machine_mem_util_percent)
            at_req_cpu_util_max.append(at_req_highest_machine_cpu_util_percent)
            at_req_mem_util_max.append(at_req_highest_machine_mem_util_percent)
            at_req_cpu_cluster_util_all.append(at_req_cluster_cpu_util_percent)
            at_req_mem_cluster_util_all.append(at_req_cluster_mem_util_percent)
            plan_cpus.append(plan_cpu)
            plan_mems.append(plan_mem)
            instance_nums.append(instance_num)

        parsed_df = pd.DataFrame({"timestamp": timestamps, "queueing_delay": q_delays, "completion_time": completion_times, "job_name": job_names, "task_name": task_names,
                                  "job_length": job_length, "cpu_avg": cpu_avgs, "cpu_max": cpu_maxes, "mem_avg": mem_avgs, "mem_max": mem_maxes,
                                  "at_req_cpu_subscribed": at_req_cpu_subscribed_all, "at_req_mem_subscribed": at_req_mem_subscribed_all,
                                  "at_req_lowest_machine_cpu_util_percent": at_req_cpu_util_min, "at_req_lowest_machine_mem_util_percent": at_req_mem_util_min,
                                  "at_req_highest_machine_cpu_util_percent": at_req_cpu_util_max, "at_req_highest_machine_mem_util_percent": at_req_mem_util_max,
                                  "at_req_cluster_cpu_util_percent": at_req_cpu_cluster_util_all, "at_req_cluster_mem_util_percent": at_req_mem_cluster_util_all,
                                  "plan_cpu": plan_cpus, "plan_mem": plan_mems, "instance_num": instance_nums})
        parsed_df.set_index("timestamp", inplace=True)
        parsed_df.to_csv(output_file, header=not os.path.exists(output_file), mode="a")
        logger.info(f"Not found percentage now {not_found / total * 100}%")
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))
    end = time.time()
    logger.info(f"grabbed qdelay to {output_file}, took {end - start}")

def calcProcessTime(starttime, cur_iter, max_iter):
    telapsed = time.time() - starttime
    testimated = (telapsed / cur_iter) * (max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).astimezone(timezone("US/Pacific")).strftime("%H:%M:%S")  # in time

    lefttime = testimated - telapsed  # in seconds

    return int(telapsed), int(lefttime), finishtime
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

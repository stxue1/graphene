import argparse
import csv
import sys
from collections import deque
from typing import List

import numpy as np
import regex as re

import pandas as pd
import logging

from dag import TaskDAG, TaskInstance, generate_task_args, JobDAG
from scheduler_job import Job
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logging.getLogger('matplotlib').setLevel(logging.WARNING)

def look_for_empty_string(options: argparse.Namespace) -> None:

    trace_file = options.trace_file
    a = None

    with open(trace_file, 'r') as f:
        reader = csv.reader(f)
        # next(reader)  # skip header

        for id, row in enumerate(reader, 1):  # Use enumerate to get the row index (1-based)
            # Start Time,End Time,Core,Memory,Priority,Lifetime  => azure2017, 2019, alibaba
            # vmcreated, vmdeleted, vmcorecount, RSS, _, _ = row
            task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem = row
            # vmcreated,vmdeleted,vmcorecount,RSS,_=row
            logger.debug("id: %d row: %s" % (id, row))
            float_core = float(plan_cpu)
            float_rss = float(plan_mem)
            a = (float_core, float_rss)

def look_for_task_name(options: argparse.Namespace) -> None:
    trace_file = options.trace_file
    max_rows = options.max or float('inf')

    regex_str = r"\D1(\D|$)"
    regex_compiled = re.compile(regex_str)

    with open(trace_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for id, row in enumerate(reader, 1):  # Use enumerate to get the row index (1-based)
            # Start Time,End Time,Core,Memory,Priority,Lifetime  => azure2017, 2019, alibaba
            vmcreated, vmdeleted, vmcorecount, RSS, priority, _, job_name, task_name = row
            # if task_name.startswith("task") or task_name.startswith("R") or task_name.startswith("M") or task_name.startswith("J") or task_name.startswith("L"):
            #     continue
            if regex_compiled.search(task_name) is None:
                continue
            print(task_name)
            if id > max_rows:
                break

def count_memory(options: argparse.Namespace) -> None:
    trace_file = options.trace_file

    total_mem = 0
    with open(trace_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for id, row in enumerate(reader, 1):  # Use enumerate to get the row index (1-based)
            # Start Time,End Time,Core,Memory,Priority,Lifetime  => azure2017, 2019, alibaba
            vmcreated, vmdeleted, vmcorecount, RSS, instance_num, lifetime, job_name, task_name = row
            if RSS == "":
                continue
            total_mem += float(RSS) * int(float(instance_num))
    print(total_mem)

def core_is_int(options: argparse.Namespace) -> None:
    trace_file = options.trace_file

    with open(trace_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for id, row in enumerate(reader, 1):  # Use enumerate to get the row index (1-based)
            # Start Time,End Time,Core,Memory,Priority,Lifetime  => azure2017, 2019, alibaba
            vmcreated, vmdeleted, vmcorecount, RSS, instance_num, lifetime, job_name, task_name = row
            if RSS == "":
                continue
            if float(int(float(vmcorecount))) != float(vmcorecount):
                print(vmcreated, vmdeleted, vmcorecount, RSS, instance_num, lifetime, job_name, task_name)

def max_cores_memory_task(options: argparse.Namespace) -> None:
    trace_file = options.trace_file
    data = pd.read_csv(trace_file)
    data['Core'] = pd.to_numeric(data['Core'], errors='coerce')
    data['Memory'] = pd.to_numeric(data['Memory'], errors='coerce')
    data = data.dropna(subset=['Core', 'Memory'])

    data['Core'] = np.where(True, data['Core'] * data['Instance Num'], data['Core'])
    data['Memory'] = np.where(True, data['Memory'] * data['Instance Num'], data['Memory'])

    max_job_memory_idx = data['Memory'].idxmax()
    max_job_cores_idx = data['Core'].idxmax()
    print(f"Maximum job memory: {data['Memory'].max()}")
    print(f"{data.iloc[max_job_memory_idx]}")
    print(f"Maximum job cores: {data['Core'].max()}")
    print(f"{data.iloc[max_job_cores_idx]}")
def max_cores_memory(options: argparse.Namespace) -> None:
    trace_file = options.trace_file
    # steal some code again

    data = pd.read_csv(trace_file)
    data['Core'] = pd.to_numeric(data['Core'], errors='coerce')
    data['Memory'] = pd.to_numeric(data['Memory'], errors='coerce')
    data = data.dropna(subset=['Core', 'Memory'])

    # alibaba
    # coalesce the "tasks" into one "job" by multiplying quantity by the number of requested resources
    # so multiply the cores and memory (time/lifetime are unaffected)
    data['Core'] = np.where(True, data['Core'] * data['Instance Num'], data['Core'])
    data['Memory'] = np.where(True, data['Memory'] * data['Instance Num'], data['Memory'])

    # Process the data for utilization calculation
    start_data = data[['Start Time', 'Core', 'Memory']].copy()
    start_data.columns = ['timestamp', 'cores', 'rss']
    start_data['event_type'] = 'start'

    end_data = data[['End Time', 'Core', 'Memory']].copy()
    end_data.columns = ['timestamp', 'cores', 'rss']
    end_data['event_type'] = 'end'

    combined_data = pd.concat([start_data, end_data]).sort_values('timestamp')

    # combined_data.loc[:, ["rss"]] = combined_data.loc[:, ["rss"]].div(4023)  # faster
    # combined_data.loc[:, ["cores"]] = combined_data.loc[:, ["cores"]].div(4023 * 96)  # faster

    cores: List[float] = []
    rss: List[float] = []
    timestamps = []
    current_cores: float = 0.0
    current_rss: float = 0.0

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

    print(f"max cores: {max(cores)}")
    print(f"max memory: {max(rss)}")

def config_cores_memory(options: argparse.Namespace) -> None:
    trace_file = options.trace_file

    vms = pd.read_csv(trace_file)
    total_cpus = vms["CPU"].sum()
    total_mem = vms["Memory"].sum()
    print(f"config total cpus: {total_cpus}")
    print(f"config total mem: {total_mem}")

def count_dag_tasks(options: argparse.Namespace) -> None:
    trace_file = options.trace_file
    # df = data.groupby(["Job Name"])["Instance Num"].sum()
    # steal some code to run some statistics
    jobs = dict()
    prev_arrival_time = -1
    vms = pd.read_csv(trace_file)
    vms_np = vms.to_numpy()

    for (i, entry) in enumerate(vms_np, 1):
        start_time, end_time, core, memory, instance_num, lifetime, job_name, task_name = entry
        # wid seems to be row index
        task = generate_task_args(task_name=task_name, job_name=job_name, instance_num=instance_num, wid=i,
                                  arrival_time=start_time, end_time=end_time, cpu=core, rss=memory, lifetime=lifetime)
        if jobs.get(job_name) is not None:
            jobs[job_name].add_dependency([task])
        else:
            jobs[job_name] = JobDAG(task.job_name, tasks=[task])
    # remove invalid job dags
    remove_keys = []
    to_delete = 0
    total = len(jobs)
    current_count = 0
    for job_name, job in jobs.items():
        current_count += 1
        if not job.valid_graph():
            to_delete += 1
            percent = 1 - to_delete / current_count
            # logger.info(f"jobname: {job.job_name}, tasks: {[task.task_name for task in job.tasks]}, now {percent:.4f}% invalid")
            remove_keys.append(job_name)

    for i in remove_keys:
        del jobs[i]
    bar_graph = []
    for job_name, job in jobs.items():
        total_instances = 0
        for task in job.tasks:
            total_instances += task.instances
        bar_graph.append(total_instances)
    s = pd.Series(bar_graph)
    a = s.value_counts()
    x = a.index
    y = a.values
    # ax = s.value_counts().plot.bar(x="task_instances per job", y="num of occurrences")
    plt.scatter(x, y)
    # ax.xaxis.set_visible(True)
    # ax.yaxis.set_visible(True)
    # fig = plt.get_figure()
    plt.savefig('task_instances_per_job.pdf')

def count_instances(options: argparse.Namespace) -> None:
    trace_file = options.trace_file
    instances = 0
    with open(trace_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for id, row in enumerate(reader, 1):  # Use enumerate to get the row index (1-based)
            # Start Time,End Time,Core,Memory,Priority,Lifetime  => azure2017, 2019, alibaba
            vmcreated, vmdeleted, vmcorecount, RSS, instance_num, lifetime, job_name, task_name = row
            if RSS == "":
                continue
            instances += int(instance_num)
    print(instances)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", "-t", help="The trace file to analyze")
    parser.add_argument("--operation", "-o", required=True, help="What test to run.")
    parser.add_argument("--max", type=int)

    options = parser.parse_args(args)

    if options.operation == "empty_string":
        look_for_empty_string(options)
    elif options.operation == "look":
        look_for_task_name(options)
    elif options.operation == "count_memory":
        count_memory(options)
    elif options.operation == "count_dag":
        count_dag_tasks(options)
    elif options.operation == "count_instances":
        count_instances(options)
    elif options.operation == "max_cores_memory":
        max_cores_memory(options)
    elif options.operation == "max_cores_memory_task":
        max_cores_memory_task(options)
    elif options.operation == "core_is_int":
        core_is_int(options)
    elif options.operation == "config":
        config_cores_memory(options)

if __name__ == "__main__":
    main()

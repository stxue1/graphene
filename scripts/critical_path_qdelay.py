import argparse
import sys
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

import logging

from batch_instance_analyze import calcProcessTime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def process_critical(options: argparse.Namespace):
    # Load the CSV file
    qdelay_file = os.path.join("batch_instance_analysis", "queueing_delays_dag.csv")
    assert os.path.exists(qdelay_file)

    output_dir = "critical_path_analysis"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_points = options.num_points or 1_000_000
    cache_file = os.path.join(output_dir, f"queueing_delays_critical_{num_points}.csv")

    if os.path.exists(cache_file) and options.overwrite is False:
        return

    columns = ["timestamp", "queueing_delay", "completion_time", "job_name", "task_name", "job_length"]

    ratios_queueing_delay_over_completion: list = []
    ratios_total_queueing_delay_over_critical: list = []
    completion_times = []
    qdelay_times = []
    job_lengths = []
    start_times = []
    end_times = []
    plan_cpus = []
    plan_mems = []
    invalid = 0
    total = 0

    start = time.time()

    num_lines = 1_228_679_841
    chunksize = 10_000_000
    max_iter = num_lines // chunksize + 1

    early_stop_iter = 1
    # job_to_tasks = get_job_to_task_mapping()
    for j, data in enumerate(pd.read_csv(qdelay_file, chunksize=chunksize)):
        # df = pd.read_csv(qdelay_file, header=None, names=columns)
        if j > early_stop_iter:
            break
        data.dropna(inplace=True)

        # filter out invalid values
        # todo: will this cause missing task dependencies?
        data = data[(data['queueing_delay'] >= 0) & (data['completion_time'] >= 0)]

        # todo: temp
        # early stop for efficiency
        # min_timestamp = 86400 * 2
        max_timestamp = 86400 * 3
        if data["timestamp"].min() > max_timestamp:
            break

        # Group tasks by job_name
        grouped = data.groupby('job_name')

        for job_name, group in grouped:
            is_dag = True

            # Check if the group contains any non-DAG tasks
            for task in group['task_name']:
                if 'task_' in task:
                    is_dag = False
                    break

            if not is_dag:
                continue

            toggle_flag = True
            if toggle_flag:
                # mapping of task to data
                task_to_data = {}

                # The file holds qdelay over instances, so calculate the max qdelay
                # or last finish to earliest start
                # The timestamp field or the first one represents the start time of the instance
                # negative/invalid values should have been sorted out above
                grouped_tasks = group.groupby('task_name')
                earliest_scheduleable_time = float('inf')
                latest_start = -float('inf')
                last_completion_time = -float('inf')
                earliest_start_time = float('inf')
                for task_name, group_of_instances in grouped_tasks:
                    current_task = None
                    for row in group_of_instances.to_numpy():
                        (timestamp, qdelay, ctime, job, task, job_length, cpu_avg, cpu_max, mem_avg, mem_max,
                         at_req_cpu_subscribed,at_req_mem_subscribed,at_req_lowest_machine_cpu_util_percent,
                         at_req_lowest_machine_mem_util_percent,at_req_highest_machine_cpu_util_percent,at_req_highest_machine_mem_util_percent,
                         at_req_cluster_cpu_util_percent,at_req_cluster_mem_util_percent,plan_cpu,plan_mem,instance_num)\
                            = row
                        # get back the scheduled time
                        # todo: make the fields exist in the original csv instead of calculating it out
                        parts = task.split('_')
                        child_task = f'task{parts[0][1:]}'

                        task_to_data.setdefault(child_task, {"earliest_start_time": earliest_start_time, "earliest_scheduleable_time": earliest_scheduleable_time,
                                                             "latest_start": latest_start, "last_completion_time": last_completion_time,
                                                             "task_name": task_name, "job_name": job_name, "cpu_avg": 0, "cpu_max": cpu_max, "mem_avg": 0,
                                                             "mem_max": mem_max, "cpu_avg_weighted": 0, "mem_avg_weighted": 0,
                                                             "count": 0, "cpu_total": 0, "mem_total": 0, "cpu_weighted_total": 0, "mem_weighted_total": 0, "total_weights": 0,   # average calculation values
                                                             "plan_cpu": 0, "plan_mem": 0, "instance_num": instance_num
                                                             }
                                                )
                        scheduleable_time = timestamp - qdelay
                        current_task = task_to_data[child_task]
                        if scheduleable_time < current_task["earliest_scheduleable_time"]:
                            current_task["earliest_scheduleable_time"] = scheduleable_time
                        if timestamp > current_task["latest_start"]:
                            current_task["latest_start"] = timestamp
                        finish_time = ctime + timestamp
                        if finish_time > current_task["last_completion_time"]:
                            current_task["last_completion_time"] = finish_time
                        if timestamp < current_task["earliest_start_time"]:
                            current_task["earliest_start_time"] = timestamp
                        current_task["count"] += 1
                        current_task["cpu_total"] += cpu_avg
                        current_task["mem_total"] += mem_avg
                        if cpu_max > current_task["cpu_max"]:
                            current_task["cpu_max"] = cpu_max
                        if mem_max > current_task["mem_max"]:
                            current_task["mem_max"] = mem_max
                        # todo: avg of avgs is not a good metric, is weight by execution time/job length sufficient?
                        current_task["cpu_weighted_total"] += cpu_avg * job_length
                        current_task["mem_weighted_total"] += mem_avg * job_length
                        # note: weights for cpu and mem are the same
                        current_task["total_weights"] += job_length

                        # sum up the plan_cpu and plan_mem, this means planned resources is the sum of all instances
                        current_task["plan_cpu"] += plan_cpu
                        current_task["plan_mem"] += plan_mem
                    if current_task is not None:
                        # the below variable will think its not defined if group_of_instances is empty, I don't think this is technically an issue
                        current_task["cpu_avg"] = current_task["cpu_total"] / current_task["count"]
                        current_task["mem_avg"] = current_task["mem_total"] / current_task["count"]
                        # I still would need to figure out the best way to deal with divide by 0 error, when the task runs immediately for 0 seconds
                        current_task["cpu_avg_weighted"] = current_task["cpu_weighted_total"] / max(current_task["total_weights"], 1)
                        current_task["mem_avg_weighted"] = current_task["mem_weighted_total"] / max(current_task["total_weights"], 1)
                # note: earliest_start_time here means the max completion time represents the
                # difference between the last finishing instance's end time and the first
                # starting instance's start time
                # so it represents any time where *something* was running, not all

                # Create the directed graph
                G = nx.DiGraph()


                def get_delay_and_time(task_data):
                    # completion last complete - earliest scheduleable, essentially realtime
                    # qdelay last task start - earliest scheduleable
                    # maximized_completion_time = task_data["last_completion_time"] - task_data["earliest_start_time"]
                    # maximized_qdelay = task_data["latest_start"] - task_data["earliest_scheduleable_time"]
                    # maximized_realtime = task_data["last_completion_time"] - task_data["earliest_scheduleable_time"]
                    maximized_completion_time = task_data["last_completion_time"] - task_data["earliest_scheduleable_time"]
                    maximized_qdelay = task_data["latest_start"] - task_data["earliest_scheduleable_time"]
                    # time spent where we first started running to when we finished
                    maximized_job_length = task_data["last_completion_time"] - task_data["earliest_start_time"]
                    return maximized_completion_time, maximized_qdelay, maximized_job_length


                for task, task_data in task_to_data.items():
                    maximized_completion_time, maximized_qdelay, maximized_job_length = get_delay_and_time(task_data)
                    parts = task_data["task_name"].split('_')
                    G.add_node(task, queueing_delay=maximized_qdelay, completion_time=maximized_completion_time, job_length=maximized_job_length,
                               cpu_avg=task_data["cpu_avg"], mem_avg=task_data["mem_avg"], cpu_avg_weighted=task_data["cpu_avg_weighted"],
                               mem_avg_weighted=task_data["mem_avg_weighted"], cpu_max=task_data["cpu_max"], mem_max=task_data["mem_max"],
                               start_time=task_data["earliest_start_time"], end_time=task_data["last_completion_time"],
                               plan_cpu=task_data["plan_cpu"], plan_mem=task_data["plan_mem"],
                               instance_num=task_data["instance_num"])
                    if len(parts) > 1:
                        for parent in parts[1:]:
                            parent_task = f'task{parent}'
                            # logger.info(f"parent: {parent_task} entire task {task_data['task_name']} and job {job_name}")
                            # parent_task_maximized_completion_time, parent_task_maximized_qdelay, parent_maximized_realtime = get_delay_and_time(task_to_data[parent_task])
                            if task_to_data.get(parent_task) is None:
                                # todo: investigate why instance does not exist but task does
                                continue
                            parent_task_maximized_completion_time, parent_task_maximized_qdelay, parent_task_maximized_job_length = get_delay_and_time(task_to_data[parent_task])
                            # todo: this only works if tasks are generated in order i think
                            G.add_edge(parent_task, task, weight=parent_task_maximized_completion_time)
                total += 1
                if not nx.is_directed_acyclic_graph(G):
                    invalid += 1
                    continue
            else:

                # Create the directed graph
                G = nx.DiGraph()

                # Extract task names, durations, and build relationships
                for _, row in group.iterrows():
                    task = row['task_name']
                    qdelay = row["queueing_delay"]
                    completion_time = row["completion_time"]
                    parts = task.split('_')

                    child_task = f'task{parts[0][1:]}'
                    G.add_node(child_task, queueing_delay=qdelay, completion_time=completion_time)

                    if len(parts) > 1:
                        for parent in parts[1:]:
                            parent_task = f'task{parent}'
                            G.add_edge(parent_task, child_task, weight=qdelay)

                total += 1

                if not nx.is_directed_acyclic_graph(G):
                    invalid += 1
                    continue

            # Calculate the critical path
            critical_path = nx.dag_longest_path(G, weight='weight')

            # calculate total queueing delay for all tasks
            # note: how useful is this measurement actually?
            total_queueing_delay = sum(nx.get_node_attributes(G, 'queueing_delay', 0).values())

            critical_path_tasks = set(critical_path)
            # calculate completion time for tasks in critical path
            critical_path_completion_time = sum(G.nodes[task].get('completion_time', 0) for task in critical_path_tasks)

            # Calculate queueing delay for tasks in critical path
            critical_path_qdelay = sum(G.nodes[task].get('queueing_delay', 0) for task in critical_path_tasks)

            # Calculat ejob length for tasks in critical path, or execution time
            critical_path_job_length = sum(G.nodes[task].get('job_length', 0) for task in critical_path_tasks)

            # Grab start time for the critical path, this will equal to when the first task is scheduled
            # I'm not sure how useful this metric will be though, since it is hard to repr long running jobs with the first task's start time
            # however, I need something related to time to match up cluster utilization information
            # maybe as an improvement, I can calculate average cluster utilization?
            critical_path_start_time = min(G.nodes[task].get('start_time') for task in critical_path_tasks)
            # note: end_time is *technically* redundant with completion_time
            critical_path_end_time = max(G.nodes[task].get('end_time') for task in critical_path_tasks)

            critical_path_plan_cpu = sum(G.nodes[task].get('plan_cpu', 0) for task in critical_path_tasks)
            critical_path_plan_mem = sum(G.nodes[task].get('plan_mem', 0) for task in critical_path_tasks)

            # the sum of completion and queueing delay should be the realtime taken // no longer true
            # critical_path_realtime = critical_path_completion_time + critical_path_qdelay

            # Calculate the ratios
            # if critical_path_realtime == 0:
            #     logger.info(f"no queueing delay or completion time detected: {critical_path_tasks} in {G.nodes}")

            # note: I dont think this is actually that useful
            if critical_path_qdelay > 0:
                ratio_queueing_delay_not_in_critical_path = total_queueing_delay / critical_path_qdelay
                ratios_total_queueing_delay_over_critical.append(ratio_queueing_delay_not_in_critical_path)
            if critical_path_completion_time > 0:
                ratio_queueing_delay_over_completion_time_in_critical_path = critical_path_qdelay / critical_path_completion_time
                ratios_queueing_delay_over_completion.append(ratio_queueing_delay_over_completion_time_in_critical_path)

            completion_times.append(critical_path_completion_time)
            qdelay_times.append(critical_path_qdelay)
            job_lengths.append(critical_path_job_length)

            start_times.append(critical_path_start_time)
            end_times.append(critical_path_end_time)
            plan_cpus.append(critical_path_plan_cpu)
            plan_mems.append(critical_path_plan_mem)
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    logger.info(f"Threw away a total of {invalid} jobs out of {total}")

    # note: this df represents stats over the entire job (multiple tasks), therefore plan_cpu is the number of requested cpus over the entire job
    df = pd.DataFrame({"start_time": start_times, "end_times": end_times, "completion_time": completion_times, "queueing_delay": qdelay_times, "job_length": job_lengths, "plan_cpu": plan_cpus, "plan_mem": plan_mems})

    df.to_csv(cache_file, index=False)

    df = pd.DataFrame({"ratios_total_queueing_delay_over_critical": ratios_total_queueing_delay_over_critical})
    # df2 = pd.DataFrame({"ratios_queueing_delay_over_completion": ratios_queueing_delay_over_completion})
    cache_file = os.path.join(output_dir, f"queueing_delays_critical_{num_points}_ratios.csv")
    df.to_csv(cache_file)

def plot_graphs(options: argparse.Namespace):
    num_points = options.num_points or 1_000_000
    output_dir = "critical_path_analysis"
    cache_file = os.path.join(output_dir, f"queueing_delays_critical_{num_points}.csv")
    df = pd.read_csv(cache_file)
    cache_file = os.path.join(output_dir, f"queueing_delays_critical_{num_points}_ratios.csv")
    df_ratios = pd.read_csv(cache_file)

    from graph_queueing_delay import plot_scatter_with_job_length
    
    # Plot the CDFs for the two requested ratios
    # Total queuing delay over critical queueing delay, how much of a job's queueing delay was on the critical path
    plot_cdf_log_x(df_ratios["ratios_total_queueing_delay_over_critical"], 'Ratio of total qdelay over log critical qdelay', 'CDF', 'CDF of total qdelay over log critical qdelay',
                   os.path.join(output_dir, f"total_queueing_delay_over_critical.png"))
    # Critical path queueing delay over the completion time
    # Comparison of time spent waiting over time spent running per job
    # plot_cdf_log_x(df_ratios["ratios_queueing_delay_over_completion"], 'Ratio of critical qdelay over log critical completion time', 'CDF',
    #                'CDF of critical qdelay over log critical completion', os.path.join(output_dir, f"critical_queueing_delay_over_critical_completion.png"))
    
    output = os.path.join(output_dir, "critical_queueing_delay_over_completion.png")
    plot_scatter_with_job_length(df["completion_time"], df["queueing_delay"], df["job_length"], xlabel="Completion Time in seconds (log)", ylabel="Queueing Delay in seconds (log)", output=output, title="Queueing Delay over Completion Time of Critical Path Instances")

# Function to plot CDF
def plot_cdf(data, xlabel, ylabel, title, output):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure(figsize=(12, 8))
    plt.plot(sorted_data, yvals, marker='.', linestyle='none')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.savefig(output)
    plt.show()


def plot_cdf_log_x(data, xlabel, ylabel, title, output):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure(figsize=(12, 8))
    # plt.plot(sorted_data, yvals, marker='.', linestyle=None)
    plt.plot(sorted_data, yvals, linestyle='solid', drawstyle='steps')
    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid(True, which="both", ls="--")

    # Set x-axis to logarithmic scale
    plt.xscale('log')

    plt.title(title, fontsize=24)  # Added title for clarity
    plt.savefig(output)
    plt.show()


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Run a specified operation.")
    parser.add_argument("--overwrite", action="store_true", default=-False)
    parser.add_argument("--num-points", "-n", type=int, default=1_000_000)
    options = parser.parse_args(args)

    process_critical(options)
    plot_graphs(options)

if __name__ == "__main__":
    main()

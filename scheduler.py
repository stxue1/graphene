from __future__ import annotations

import argparse
import functools
import pickle
from typing import Any, Optional

import coloredlogs
import numpy as np
import os
import sys
import pandas as pd

import networkx as nx
import copy
import heapq
import time
import datetime as dt
from itertools import islice

from matplotlib import pyplot as plt
from pytz import timezone
from collections import deque
import logging

from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

coloredlogs.install(level=logger.getEffectiveLevel(), logger=logger)

"""

DAGs in this script are per instance (equivalent of tasks in the paper)

DAG nodes have the names `task_name/instance_number` where "/" is the delimiter

"""


def calcProcessTime(starttime, cur_iter, max_iter):
    telapsed = time.time() - starttime
    testimated = (telapsed / cur_iter) * (max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).astimezone(timezone("US/Pacific")).strftime("%H:%M:%S")  # in time

    lefttime = testimated - telapsed  # in seconds

    return int(telapsed), int(lefttime), finishtime


def calculate_fragscore(resources: list[str], dag_allocation: dict[str, float], tasks_duration: list[int], tasks_demands: dict[str, list[float]], exec_time: int) -> float:
    """
    Reflects packability of tasks in a stage

    Total work in stage / How long greedy packer takes to schedule

    It seems like greedy packer time is set to ExecTime(s) where s is a stage
    So it should be the estimated time to execute all tasks in s

    Essentially, total work / total exec time

    TWork: total work in DAG normalized by cluster share of DAG
    (no dep, perfect packing, job will finish in twork)

    It seems like work is defined as duration * resource demands

    This is my best interpretation of TWork:
    Max for each resource (cpu, mem) of: sum of task duration * task resource demands (cpu fraction, mem fraction) divided by fraction of resource allocated to DAG

    Both requests and allocated should be fractions, as we are trying to get 0<=TWork<=1
    A key note: resource allocated to DAG seems to represent the resources the scheduler allows the DAG to use, not the requested resource or the actual use
    For Alibaba, I don't think I have a good metric of this resource allocation as I don't have the scheduler inner workings
    todo: Determine if this metric is calculated inside the scheduler and/or within the Microsoft trace, from preliminary observations, it seems to be internal to the scheduler
    :param resources: list of resources to consider as string
    :param dag_allocation: dictionary of resources to allocated resources
    :param tasks_duration: list of duration of tasks
    :param tasks_demands: dictionary of resources to list of resource demands
    :param exec_time: estimated execution time.
    """
    # resources = ['cpu', 'mem']
    # dag_allocation = {'cpu': 0.5, 'mem': 0.5}
    # tasks_duration = [1, 4, 6, 8]
    # tasks_demands = {'cpu': [0.1, 0.1, 0.2, 0.2], 'mem': [0.2, 0.1, 0.1, 0.3]}

    # I'm not sure if estimated execution time can be substituted with the actual running time in alibaba, as it should represent execution time in the simulation
    # Paper says they ran jobs once and recorded observed usage
    t_work = max([1 / dag_allocation[resource] * np.dot(tasks_duration, tasks_demands[resource]) for resource in resources])
    # logger.info(f"calculated twork {t_work} with exec time {exec_time}")
    return t_work / max(1, exec_time)


def calculate_longscore(task_duration: int, max_task_duration: int):
    """
    Divides task duration by the maximum task duration

    Tasks with a higher score are more likely to be on the critical path
    :param task_duration: task duration to consider
    :param max_task_duration: max task duration in graph (calculate before for efficiency)
    """
    # this function is mostly for readability then
    return task_duration / max_task_duration


def closure(graph: nx.DiGraph, considered_nodes: set) -> set:
    # I have two options: create a transitive closure and then iterate through all considered nodes or iterate through app pairs in the considered nodes and grab all nodes
    # in between in all/any path (i think, since we need all paths to be satisfied for dependency reasons)
    # the first option is O(V^3) where V is the vocab of the graph or number of nodes, see:
    # https://algowiki-project.org/en/Transitive_closure_of_a_directed_graph
    # https://stackoverflow.com/a/59349838
    # https://en.wikipedia.org/wiki/Shortest_path_problem#All-pairs_shortest_paths (it's probably not worth implementing a custom solver, especially in python)
    # the second option is O(n^2C) where n is the number of considered nodes in the graph and C is the amortized cost to look for all paths
    # I believe the second option is better as n can only be so large, though considering networkx is slow, O(C) may still be significant
    closure = set(considered_nodes)
    for inode in considered_nodes:
        for jnode in considered_nodes:
            if inode == jnode:
                continue
            try:
                # todo: Given the task names, it may be possible to determine which order is correct
                all_paths = nx.all_simple_paths(graph, source=inode, target=jnode)
                for path in all_paths:
                    closure.update(path)

                rev_all_paths = nx.all_simple_paths(graph, source=jnode, target=inode)
                for path in rev_all_paths:
                    closure.update(path)
            except nx.NetworkXNoPath:
                pass
    return closure


def closure2(graph: nx.DiGraph, considered_nodes: set) -> set:
    """
    Same as the other closure operation, but use BFS to generate all reachable to and reachable from nodes, then take the intersection
    """
    if not considered_nodes:
        return set()

    queue = deque(considered_nodes)
    reachable_from_considered = set()

    visited = set()
    while queue:
        node = queue.popleft()
        for successor in graph.successors(node):
            if successor not in visited:
                visited.add(successor)
                if successor not in reachable_from_considered:
                    reachable_from_considered.add(successor)
                    queue.append(successor)

    queue2 = deque(considered_nodes)
    reachable_to_considered = set()

    visited = set()
    while queue2:
        node = queue2.popleft()
        for predecessor in graph.predecessors(node):
            if predecessor not in visited:
                visited.add(predecessor)
                if predecessor not in reachable_to_considered:
                    reachable_to_considered.add(predecessor)
                    queue2.append(predecessor)

    return reachable_to_considered.intersection(reachable_from_considered).union(considered_nodes)


def candidate_troublesome_tasks(graph: nx.DiGraph) -> set[tuple[frozenset, frozenset, frozenset, frozenset]]:
    """
    Identify the most troublesome tasks in the DAG, returns four subsets

    :param graph: DAG graph of all tasks to run in a job
    """
    subsets_L = set()
    seen = set()
    # todo: networkx is slow
    epsilon = 0.1
    epsilon_reciprocal = int(1 / epsilon)
    assert 1 / epsilon_reciprocal == epsilon

    all_tasks = graph.nodes(data=True)
    all_task_names = []
    max_duration = 0.1
    for name, details in all_tasks:
        all_task_names.append(name)
        max_current_duration = details['duration']
        max_duration = max_duration if max_duration > max_current_duration else max_current_duration

    # calculate stage demands and durations here
    # stage represents a reduce/map, since it represents the same code running over different data, I will assume
    # an alibaba task is equivalent to a graphene stage, for the code, I will still use tasks as the equivalent of alibaba instances for simplicity in translating
    # the paper into code
    stages = {}
    for name, details in all_tasks:
        # format should be task/instance_num
        stage_name = name.split('/')[0]
        stages.setdefault(stage_name, {'durations': [], 'demands': {'cpu': [], 'mem': []}})
        stages[stage_name]['durations'].append(details['duration'])
        stages[stage_name]['demands']['cpu'].append(details['cpu'])
        stages[stage_name]['demands']['mem'].append(details['mem'])

    # calculate stage frag scores
    for stage_name, stage in stages.items():
        # todo: clarify what is exec time of a stage
        # todo: is cluster share simply the sum of wanted resources, assuming we allocate it?
        summed_cpu_demands = sum(stage['demands']['cpu'])
        summed_mem_demands = sum(stage['demands']['mem'])
        if summed_cpu_demands == 0:
            # some epsilon
            summed_cpu_demands = 0.00001
        if summed_mem_demands == 0:
            summed_mem_demands = 0.00001
        dag_allocation = {"cpu": summed_cpu_demands, "mem": summed_mem_demands}
        stages[stage_name]['frag_score'] = calculate_fragscore(['cpu', 'mem'], dag_allocation,
                                                               stage['durations'], stage['demands'], max(stage['durations']))

    for i in range(0, epsilon_reciprocal, 1):
        threshold_l = epsilon * i
        for j in range(0, epsilon_reciprocal, 1):
            threshold_f = epsilon * j
            t_nodes = set()
            for name, details in all_tasks:
                long_score = calculate_longscore(details['duration'], max_duration)
                # This calculation is per stage, which to my understanding, should be per task
                stage_name = name.split('/')[0]
                frag_score = stages[stage_name]['frag_score']
                if long_score >= threshold_l or frag_score <= threshold_f:
                    t_nodes.add(name)
            t_nodes = closure2(graph, t_nodes)

            if t_nodes in seen:
                continue
            seen.add(frozenset(t_nodes))

            p_nodes = set()
            c_nodes = set()
            for node in t_nodes:
                children = set(graph.successors(node))
                c_nodes.update(children)

                parents = set(graph.successors(node))
                p_nodes.update(parents)

            p_nodes -= t_nodes
            c_nodes -= t_nodes

            s_nodes = set(all_task_names)
            s_nodes -= t_nodes
            s_nodes -= p_nodes
            s_nodes -= c_nodes

            # convert into tuples because i cant add sets into sets
            subsets_L.add((frozenset(t_nodes), frozenset(s_nodes), frozenset(p_nodes), frozenset(c_nodes)))
    return subsets_L


@functools.lru_cache(maxsize=128)
def calculate_srpt_j(running: frozenset[Machine.Task]):
    return np.sum([task.dot() for task in running])


class Machine:
    class Task:
        def __init__(self, task_name, duration, start_time, cpu, mem):
            self.task_name = task_name
            self.duration = duration
            self.cpu = cpu
            self.mem = mem
            self.start_time = start_time
            self.end_time = start_time + duration

        @property
        def demands(self):
            return self.cpu, self.mem

        @property
        def norm(self):
            return np.sqrt(self.cpu ** 2 + self.mem ** 2)

        @property
        def durations(self):
            return self.duration, self.duration

        def dot(self):
            # todo: I'm not sure if this is supposed to be norm or abs
            # return self.duration * self.norm
            return np.dot(self.durations, self.demands)

    def __init__(self, initial_cpu: float, initial_memory: float):
        """
        Initializes the resource space.

        Args:
            initial_cpu: The initial amount of CPU available.
            initial_memory: The initial amount of memory available.
        """
        self.total_cpu = initial_cpu
        self.total_memory = initial_memory
        # running tasks, mapping job name to task information
        self.running: dict[str, set[Machine.Task]] = dict()
        self.initial = initial_cpu, initial_memory

        self.usage_log: dict = dict()

    @property
    def resources(self):
        return self.total_cpu, self.total_memory

    @property
    def usage(self):
        return self.initial[0] - self.total_cpu, self.initial[1] - self.total_memory

    def log_usage(self, timestamp: int):
        self.usage_log[timestamp] = self.usage

    def calculate_perfscore(self, tpriscore: float, job_name: str, cpu: float, mem: float, eta: float):
        pscore, oscore = 0, 0
        if cpu <= self.total_cpu and mem <= self.total_memory:
            # fits
            pscore = np.dot(self.resources, (cpu, mem))
        else:
            # overbook
            pass
        srpt_j = calculate_srpt_j(frozenset(self.running.get(job_name, set())))

        perfscore = tpriscore * max(pscore, oscore) - eta * srpt_j
        return perfscore

    def argmax_perfscore_per_job(self, job_name, task_info: dict[str, tuple[Any, ...]], eta: float) -> str:
        # argmax perfscore
        max_perfscore = None
        best_task = None
        for task_name, (tpriscore, cpu, mem, duration) in task_info.items():
            perfscore = self.calculate_perfscore(tpriscore, job_name, cpu, mem, eta)
            if perfscore > (max_perfscore or float("-inf")):
                max_perfscore = perfscore
                best_task = task_name
        return best_task

    def argmax_perfscore(self, task_info: dict[str, dict[str, tuple[Any, ...]]], eta: float) -> tuple[str, str]:
        # argmax perfscore
        max_perfscore = None
        best_task = None
        best_job = None
        for job_name, task_data in task_info.items():
            for task_name, (tpriscore, cpu, mem, duration) in task_data.items():
                perfscore = self.calculate_perfscore(tpriscore, job_name, cpu, mem, eta)
                if perfscore > (max_perfscore or float("-inf")):
                    max_perfscore = perfscore
                    best_task = task_name
                    best_job = job_name
        return best_job, best_task

    def use_resource(self, cpu, mem):
        self.total_cpu -= cpu
        self.total_memory -= mem

    def free_resource(self, cpu, mem):
        self.total_cpu += cpu
        self.total_memory += mem

    def resources_are_available(self, cpu, mem):
        return self.total_cpu >= cpu and self.total_memory >= mem

    def run_task(self, job_name, task_name, cpu, mem, duration, start_time):
        """
        end_tasks should have been run sometime before this, currently, this is done by the scheduler over all machines
        """
        self.use_resource(cpu, mem)
        self.running.setdefault(job_name, set())
        self.running[job_name].add(Machine.Task(task_name, duration, start_time, cpu, mem))

    def end_tasks(self, current_time) -> dict:
        """
        Remove finished tasks and free up their resources
        """
        all_finished = dict()
        for job_name, tasks in self.running.items():
            finished = set()
            for task in tasks:
                # or maybe just greater
                if current_time >= task.end_time:
                    finished.add(task.task_name)
                    self.free_resource(task.cpu, task.mem)
            if len(finished) > 0:
                all_finished[job_name] = finished
            self.running[job_name] = set(filter(lambda t: t.task_name not in finished, tasks))
        return all_finished


class OnlineScheduler:
    def __init__(self, machines: int, cpu: float, mem: float, eta, k, factor_t, deficit_threshold):
        """
        :param machines: number of machines
        :param cpu: cpu per machine
        :param mem: mem per machine
        """
        self.eta = eta
        # store mapping of job name to deficit counter
        self.job_deficit: dict[str, int] = dict()
        # same as task_data in values
        # mapping of jobs to task data in order to store other job information while trying to schedule one job at a time
        self.job_tasks: dict[str, dict[str, tuple[Any, ...]]] = {}

        # todo: define this
        self.cluster_capacity = machines

        # finished tasks, mapping of job to set of tasks
        self.finished: dict[str, set] = dict()

        self.machines = [Machine(cpu, mem) for _ in range(machines)]

        # todo: define this
        self.factor_t = 1

        self.next_timestamps = set()

        self.timestamp_to_usage = dict()

        self.total_mem = machines * mem
        self.total_cpu = machines * cpu

        self.deficit_threshold = deficit_threshold

        # predecessor tracking to speedup
        self.remaining_predecessors: dict[str, dict[str, set[str]]] = {}
        # mapping of predecessor to task (child)
        self.dependents: dict[str, dict[str, set[str]]] = {}

        self.time = 0

    def dump_cluster_usage(self, output: str):
        timestamps = np.array(list(self.timestamp_to_usage.keys()))
        cpu = np.array(list(value[0] for value in self.timestamp_to_usage.values()))
        memory = np.array(list(value[1] for value in self.timestamp_to_usage.values()))
        df = pd.DataFrame({"timestamp": timestamps, "cpu": cpu, "mem": memory})
        df.to_csv(output, index=False)

    def plot_cluster_usage(self, filename: str, input_filename: Optional[str] = None, sigma: Optional[int] = None):
        # technically this separation of data ownership b/w input data and cluster size is bad, but itll work temporarily
        if input_filename:
            df = pd.read_csv(input_filename)
            timestamps = df["timestamp"]
            cpu = df["cpu"]
            memory = df["mem"]
        else:
            timestamps = np.array(list(self.timestamp_to_usage.keys()))
            cpu = np.array(list(value[0] for value in self.timestamp_to_usage.values()))
            memory = np.array(list(value[1] for value in self.timestamp_to_usage.values()))

        # norm
        cpu /= self.total_cpu
        memory /= self.total_mem

        if sigma:
            cpu = gaussian_filter1d(cpu, sigma=sigma)
            memory = gaussian_filter1d(memory, sigma=sigma)

        fig, ax1 = plt.subplots(figsize=(10, 4))  # plt.subplots(figsize=(9, 3))  # Set a figure size in inches

        color = 'crimson'
        ax1.set_xlabel('Time (seconds)', fontsize=28)
        ax1.set_ylabel('CPU (%)', color=color, fontsize=28)
        ax1.plot(timestamps, cpu, color=color, linewidth=3)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
        ax1.tick_params(axis='x', labelsize=20)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'slateblue'
        ax2.set_ylabel('Memory (%)', color=color, fontsize=28)  # we already handled the x-label with ax1
        ax2.plot(timestamps, memory, color=color, linewidth=3)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.title('Utilization Graph')
        plt.savefig(f'scheduler_data/{filename}', bbox_inches='tight', pad_inches=0.07)
        plt.show()

    def add_predecessors_for_job(self, job_name: str, task_name: str, predecessors: tuple[str]):
        self.remaining_predecessors.setdefault(job_name, dict())
        self.remaining_predecessors[job_name][task_name] = set(predecessors)
        self.dependents.setdefault(job_name, dict())
        for p in predecessors:
            self.dependents[job_name].setdefault(p, set())
            self.dependents[job_name][p].add(task_name)

    def remove_predecessors(self, job_name: str, predecessor_task: str):
        # there may not be any children or dependent jobs to this one
        for dependent_task in self.dependents[job_name].get(predecessor_task, []):
            self.remaining_predecessors[job_name][dependent_task].discard(predecessor_task)

    def remove_job_precedessor_tracker(self, job_name: str):
        del self.remaining_predecessors[job_name]
        del self.dependents[job_name]

    def check_predecessors(self, job_name, task_to_check):
        return len(self.remaining_predecessors[job_name][task_to_check]) == 0

    def check_predecessors_old(self, job_name, predecessors: tuple[str]):
        return all(p in self.finished.get(job_name, {}) for p in predecessors)

    def set_finished_bunch(self, job_name, tasks: set[str]):
        self.finished.setdefault(job_name, set())
        self.finished[job_name].update(tasks)
        for t in tasks:
            # todo: if we schedule everything before the tasks finish, these will already be gone
            # I should move ownership of this handling into the scheduler itself (the deletion of predecessor trackers)
            if self.dependents.get(job_name):
                self.remove_predecessors(job_name, t)

    def delete_finished(self, job_name: str):
        # I think if all tasks in a job are scheduled at once, it will never exist in the finished dictionary
        if self.finished.get(job_name):
            del self.finished[job_name]

    def find_appropriate_tasks_for_machine(self, machine: Machine, tasks: dict[str, list[tuple[str, tuple[Any, ...]]]], current_time: int) -> tuple[dict, list]:
        """
        Try and schedule all tasks onto current machine at a given timestamp

        Returns set of scheduled task names and the next timestamps to consider, which is the timestamps when a task completes

        :param tasks: new tasks to add
        :return: dictionary of job name to scheduled tasks and list of next times to consider
        """
        # todo: these ownership constructs results in a not a very pythonic (or intuitive) approach
        scheduled = dict()
        next_timestamp = []
        while True:
            for job_name, task_data in tasks.items():
                for task_name, (duration, cpu, mem, tpriscore, predecessors) in task_data:
                    # already scheduled tasks are no longer considered
                    if task_name in scheduled.get(job_name, set()):
                        continue
                    # store demands alongside
                    # todo: graphene doesnt say how they do duration here
                    self.job_tasks.setdefault(job_name, dict())
                    self.job_tasks[job_name][task_name] = (tpriscore, cpu, mem, duration)
                    self.job_deficit.setdefault(job_name, 0)

            if len(tasks) == 0:
                break

            best_job, best_task = machine.argmax_perfscore(self.job_tasks, self.eta)

            # if unfairness is too high, choose another job
            current_job = best_job

            if len(self.job_deficit) > 0:
                # pyright is too stupid to understand this?
                job_most_deficit = max(self.job_deficit, key=self.job_deficit.get)  # type: ignore
                
                highest_deficit = self.job_deficit[job_most_deficit]
                if self.deficit_threshold is not None and highest_deficit >= self.deficit_threshold:
                    assert job_most_deficit is not None
                    discriminated_task_perfscores = self.job_tasks[job_most_deficit]
                    best_task = machine.argmax_perfscore_per_job(job_most_deficit, discriminated_task_perfscores, self.eta)
                    current_job = job_most_deficit

            if best_task is None:
                break

            best_task_data = self.job_tasks[current_job][best_task]

            task_cpu = best_task_data[1]
            task_mem = best_task_data[2]
            task_duration = best_task_data[3]
            # in hindsight, maybe I could've used Space, but this in theory should be faster
            if machine.resources_are_available(task_cpu, task_mem):
                # if we scheduled a job, reset its deficit to 0
                self.job_deficit[current_job] = 0

                next_timestamp.append(task_duration + current_time)
                scheduled.setdefault(current_job, set())
                scheduled[current_job].add(best_task)
                machine.run_task(current_job, best_task, task_cpu, task_mem, task_duration, current_time)
                # update deficit counters, use slot fairness (or something similar to it)
                for j in self.job_deficit.keys():
                    if current_job != j:
                        self.job_deficit[j] += 1
                # remove since we scheduled
                del self.job_tasks[current_job][best_task]
                tasks[current_job] = list(filter(lambda t: t[0] != best_task, tasks[current_job]))
                # get rid of empty job deficits
                if len(self.job_tasks[current_job]) == 0:
                    del self.job_deficit[current_job]
                    del self.job_tasks[current_job]
            # if we can't schedule, this should mean no machines are available for this task, try again later (no overbooking implemented)
            else:
                break
        return scheduled, next_timestamp

    def finish_rest(self, current_time: int):
        max_time = current_time + 100
        # ideally this should be part of the schedule loop
        while len(self.next_timestamps) != 0:
            if max_time < min(self.next_timestamps):
                break
            self.finish()

    def finish(self):
        current_time = min(self.next_timestamps)

        for machine in self.machines:
            finished = machine.end_tasks(current_time)
            for finished_job_name, finished_tasks in finished.items():
                self.set_finished_bunch(finished_job_name, finished_tasks)
        self.timestamp_to_usage[current_time] = (0, 0)
        for machine in self.machines:
            machine_cpu_usage, machine_mem_usage = machine.usage
            self.timestamp_to_usage[current_time] = (self.timestamp_to_usage[current_time][0] + machine_cpu_usage, self.timestamp_to_usage[current_time][1] + machine_mem_usage)
            machine.log_usage(current_time)

        self.next_timestamps = set(filter(lambda x: x > current_time, self.next_timestamps))
    def check_finished(self, current_time: int):
        for machine in self.machines:
            finished = machine.end_tasks(current_time)
            for finished_job_name, finished_tasks in finished.items():
                self.set_finished_bunch(finished_job_name, finished_tasks)

    def log_usage(self, current_time: int):
        self.timestamp_to_usage[current_time] = (0, 0)
        for machine in self.machines:
            machine_cpu_usage, machine_mem_usage = machine.usage
            self.timestamp_to_usage[current_time] = (self.timestamp_to_usage[current_time][0] + machine_cpu_usage, self.timestamp_to_usage[current_time][1] + machine_mem_usage)
            machine.log_usage(current_time)

    def schedule_tasks_bunch(self, scheduleable_jobs: dict[str, list[tuple[str, tuple[Any, ...]]]], current_time: int, typ: str):
        """
        Run a bunch of jobs specified in a dictionary mapping from job name to task orderings

        Returns a list of scheduled job names and a dictionary of job name to new task orderings (after running those jobs), and the next timestamp to consider
        """
        finished_jobs = set()
        for job_name in scheduleable_jobs.keys():
            if job_name in finished_jobs:
                continue
            scheduled, next_timestamp = self.schedule_tasks_type(scheduleable_jobs, current_time, typ)

            for scheduled_job, scheduled_tasks in scheduled.items():
                # remove scheduled
                leftover_task_ordering = list(filter(lambda t: t[0] not in scheduled_tasks, scheduleable_jobs[scheduled_job]))

                # remove scheduled tasks
                # This alters the dictionary while iterating, so I must iterate over keys
                scheduleable_jobs[scheduled_job] = leftover_task_ordering

                if len(leftover_task_ordering) == 0:
                    # since i cant update a dictionary while iterating, record jobs that are finished
                    # this is necessary with fairness as other jobs that are not in the current iteration loop can be ran
                    finished_jobs.add(scheduled_job)

            if next_timestamp:
                self.next_timestamps.add(next_timestamp)

            if typ.lower() == "fifo":
                # enforce fifo order by running first job only
                break
        # remove scheduled jobs
        for job_name in finished_jobs:
            del scheduleable_jobs[job_name]
            self.delete_finished(job_name)
            self.remove_job_precedessor_tracker(job_name)
        if len(self.next_timestamps) == 0:
            # this can happen with slot fairness since once we force a job to be scheduled it is no longer guaranteed to have enough resources
            next_timestamp = current_time + 1
        else:
            next_timestamp = min(self.next_timestamps)
        self.next_timestamps = set(filter(lambda x: x > next_timestamp, self.next_timestamps))
        return scheduleable_jobs, next_timestamp

    def schedule_tasks_type(self, tasks: dict[str, list[tuple[str, tuple[Any, ...]]]], current_time: int, typ: str):
        if typ.lower() == "fifo":
            return self.schedule_tasks_fifo(tasks, current_time)
        else:
            return self.schedule_tasks(tasks, current_time)

    def schedule_tasks(self, tasks: dict[str, list[tuple[str, tuple[Any, ...]]]], current_time: int):
        """
        Schedule a set of tasks at a certain time

        Returns set of task names that were scheduled, leftover tasks that could not be scheduled, and the next timestamp to consider (first timestmap when the next task ends)
        """
        DELTA = 25
        if current_time - self.time > DELTA:
            # large time gaps can result in faulty graph representations as it doesn't log the usage prior to running an instance
            # as a quick workaround, past a certain threshold, log the timestamp right before
            # this doesnt completely resolve the issue but prevents large time gaps from warping the graph (small gaps likely won't matter too much)
            self.check_finished(current_time - 1)
            # logger.info(f'logging {current_time - 1}')
            self.log_usage(current_time - 1)
        self.time = current_time

        self.check_finished(current_time)

        total_scheduled = dict()
        for machine in self.machines:
            # scheduleable = list(filter(lambda t: self.check_predecessors_old(job_name, t[1][4]) and t[0] not in self.finished.get(job_name, {}), tasks))
            scheduleable = {job_name: list(filter(lambda t: self.check_predecessors(job_name, t[0]) and t[0] not in self.finished.get(job_name, {}), job_tasks)) for job_name, job_tasks in tasks.items()}

            scheduled, timestamps = self.find_appropriate_tasks_for_machine(machine, scheduleable, current_time)

            self.next_timestamps.update(set(timestamps))
            for j, scheduled_tasks in scheduled.items():
                total_scheduled.setdefault(j, set())
                total_scheduled[j].update(scheduled_tasks)
        self.log_usage(current_time)
        # return new set of tasks that could not be ran yet, in order to schedule later
        if len(self.next_timestamps) > 0:
            next_timestamp = min(self.next_timestamps)
            self.next_timestamps = set(filter(lambda x: x > next_timestamp, self.next_timestamps))
        else:
            # no more timestamps left, probably implies everything in the current task list has no work left, return none
            next_timestamp = None
            # next_timestamp = current_time + 1
        return total_scheduled, next_timestamp


    def find_appropriate_tasks_for_machine_fifo(self, machine: Machine, tasks: dict[str, list[tuple[str, tuple[Any, ...]]]], current_time: int) -> tuple[dict, list]:
        # this is a pretty messy but itll do the trick
        scheduled = set()
        next_timestamp = []
        total_scheduled = {}
        while True:
            best_task = None
            best_job = None
            task_cpu, task_mem, task_duration = None, None, None
            for job_name, task_data in tasks.items():
                for task_name, (duration, cpu, mem, tpriscore, predecessors) in task_data:
                    # already scheduled tasks are no longer considered
                    if task_name in scheduled:
                        continue
                    if machine.resources_are_available(cpu, mem):
                        best_task = task_name
                        best_job = job_name
                        task_cpu = cpu
                        task_mem = mem
                        task_duration = duration
                # force fifo order, do first job
                break
            if len(tasks) == 0:
                # no more tasks
                break
            current_job = best_job

            if best_task is None:
                # cannot schedule, no resources
                break

            assert task_cpu is not None
            assert task_mem is not None
            assert task_duration is not None

            next_timestamp.append(task_duration + current_time)
            scheduled.add(best_task)
            machine.run_task(current_job, best_task, task_cpu, task_mem, task_duration, current_time)
            # remove since we scheduled
            total_scheduled.setdefault(best_job, set())
            total_scheduled[current_job].add(best_task)
            tasks[current_job] = list(filter(lambda t: t[0] != best_task, tasks[current_job]))

        return total_scheduled, next_timestamp
    def schedule_tasks_fifo(self, tasks: dict[str, list[tuple[str, tuple[Any, ...]]]], current_time: int):
        # messy, but itll do the trick
        DELTA = 25
        if current_time - self.time > DELTA:
            self.check_finished(current_time - 1)
            self.log_usage(current_time - 1)
        self.time = current_time
        self.check_finished(current_time)
        total_scheduled = dict()
        for machine in self.machines:
            scheduleable = {job_name: list(filter(lambda t: self.check_predecessors(job_name, t[0]) and t[0] not in self.finished.get(job_name, {}), job_tasks)) for job_name, job_tasks in tasks.items()}
            scheduled, timestamps = self.find_appropriate_tasks_for_machine_fifo(machine, scheduleable, current_time)

            self.next_timestamps.update(set(timestamps))
            for j, scheduled_tasks in scheduled.items():
                total_scheduled.setdefault(j, set())
                total_scheduled[j].update(scheduled_tasks)
        self.log_usage(current_time)
        # return new set of tasks that could not be ran yet, in order to schedule later
        if len(self.next_timestamps) > 0:
            next_timestamp = min(self.next_timestamps)
            self.next_timestamps = set(filter(lambda x: x > next_timestamp, self.next_timestamps))
        else:
            next_timestamp = None
        return total_scheduled, next_timestamp

    @property
    def cluster_usage(self):
        cpu = 0
        mem = 0
        for machine in self.machines:
            machine_cpu, machine_mem = machine.resources
            cpu += machine_cpu
            mem += machine_mem
        return cpu, mem


class Space:
    """
    Represents the resource space of CPU, memory, and time, allowing for
    tracking resource availability over time.
    """

    def __init__(self, initial_cpu: int, initial_memory: int):
        """
        Initializes the resource space.

        :param initial_cpu: The initial amount of CPU available.
        :param initial_memory: The initial amount of memory available.
        """
        self.total_cpu = initial_cpu
        self.total_memory = initial_memory
        self.cpu_availability: dict[int, int] = {0: initial_cpu}  # {timestamp: available_cpu}
        self.memory_availability: dict[int, int] = {0: initial_memory}  # {timestamp: available_memory}
        self.cpu_usage_timeline: list[tuple[int, int, int]] = []  # (start_time, end_time, used_cpu) - min-heap based on end_time
        self.memory_usage_timeline: list[tuple[int, int, int]] = []  # (start_time, end_time, used_memory) - min-heap based on end_time
        self.placed: dict[str, tuple[Any, ...]] = {}  # {task_name: (start_time, end_time)}

    def insert_task(self, name: str, cpu_needed: int, memory_needed: int, duration: int, start_time: int = 0) -> bool:
        """
        Attempts to insert a task into the resource space.

        :param cpu_needed: The amount of CPU required by the task.
        :param memory_needed: The amount of memory required by the task.
        :param duration: The duration for which the task will run (in seconds).
        :param start_time: The earliest time the task can start (defaults to 0).

        :return: True if the task can be scheduled, False otherwise.
        """
        available_cpu_time = self._find_earliest_available(self.cpu_availability, cpu_needed, self.cpu_usage_timeline, start_time)
        available_memory_time = self._find_earliest_available(self.memory_availability, memory_needed, self.memory_usage_timeline, start_time)

        schedule_time = max(available_cpu_time, available_memory_time)
        end_time = schedule_time + duration

        if self.get_available_cpu(schedule_time) >= cpu_needed and \
                self.get_available_memory(schedule_time) >= memory_needed:
            # Update CPU availability
            self._update_availability(self.cpu_availability, schedule_time, -cpu_needed)
            self._update_availability(self.cpu_availability, end_time, cpu_needed)
            heapq.heappush(self.cpu_usage_timeline, (schedule_time, end_time, cpu_needed))

            # Update Memory availability
            self._update_availability(self.memory_availability, schedule_time, -memory_needed)
            self._update_availability(self.memory_availability, end_time, memory_needed)
            heapq.heappush(self.memory_usage_timeline, (schedule_time, end_time, memory_needed))

            # For record keeping
            self.placed[name] = (schedule_time, end_time, cpu_needed, memory_needed, duration)

            return True
        else:
            # Recurisvely try again with a new timestamp
            # this can happen if at the next timestamp, the other resource is not available
            return self.insert_task(name, cpu_needed, memory_needed, duration, start_time=schedule_time)

    def end_time(self, task_name: str) -> int:
        """
        Get the end time of a placed task. Task must have already been placed.
        """
        return self.placed[task_name][1]

    def completion_time(self) -> int:
        """
        Get the runtime of the entire DAG. DAG must already be placed.
        """
        # This should equal the last end_time
        return max(timestamp[1] for timestamp in self.placed.values())

    def invert(self):
        """
        Invert the space by inverting all timestamps. Use only when converting backwards placements into forward placements.
        """
        if len(self.placed) == 0:
            # nothing to invert
            return
        completion_time = self.completion_time()
        for name, timestamps in self.placed.items():
            self.placed[name] = (completion_time - timestamps[0], completion_time - timestamps[1]) + tuple(timestamps[2:])

    def order_tasks(self):
        # I'm not sure if order includes schedule time or a ranking
        return sorted(list(self.placed.items()), key=lambda x: x[1][0])

    def placed_tasks(self):
        return [t for t in self.placed.keys()]

    def task_details(self, graph: nx.DiGraph):
        """
        Return tuple of task name to details, including dependencies
        """
        ordered = self.order_tasks()
        processed = []
        num_tasks = len(ordered)
        for i, (task_name, task_info) in enumerate(ordered):
            duration = task_info[4]
            cpu = task_info[2]
            mem = task_info[3]
            # this might be a little slow due to networkx
            predecessors = tuple(graph.predecessors(task_name))
            # + 1 to avoid negative calculations later
            processed.append((task_name, (duration, cpu, mem, (num_tasks - i) / num_tasks, predecessors)))
        return processed

    def get_available_cpu(self, timestamp: int) -> int:
        """
        Returns the amount of CPU available at a given timestamp.
        """
        return self._get_available_resource(self.cpu_availability, self.cpu_usage_timeline, self.total_cpu, timestamp)

    def get_available_memory(self, timestamp: int) -> int:
        """
        Returns the amount of memory available at a given timestamp.
        """
        return self._get_available_resource(self.memory_availability, self.memory_usage_timeline, self.total_memory, timestamp)

    def get_earliest_available_cpu(self, cpu_needed: int, after_timestamp: int = 0) -> int:
        """
        Finds the earliest timestamp after the given timestamp when the
        specified amount of CPU will be available.
        """
        return self._find_earliest_available(self.cpu_availability, cpu_needed, self.cpu_usage_timeline, after_timestamp)

    def get_earliest_available_memory(self, memory_needed: int, after_timestamp: int = 0) -> int:
        """
        Finds the earliest timestamp after the given timestamp when the
        specified amount of memory will be available.
        """
        return self._find_earliest_available(self.memory_availability, memory_needed, self.memory_usage_timeline, after_timestamp)

    def _get_available_resource(self, availability_map: dict[int, int], usage_timeline: list[tuple[int, int, int]], total_resource: int, timestamp: int) -> int:
        """
        Helper function to get the available amount of a resource at a given timestamp.
        """
        # Efficiently prune finished tasks from the timeline
        while usage_timeline and usage_timeline[0][1] <= timestamp:
            _, end_time, used = heapq.heappop(usage_timeline)
            self._update_availability(availability_map, end_time, used)

        # Find the latest availability recorded before or at the timestamp
        sorted_times = sorted(availability_map.keys())
        latest_available_time = 0
        for t in sorted_times:
            if t <= timestamp:
                latest_available_time = t
            else:
                break
        return availability_map.get(latest_available_time, total_resource)

    def _find_earliest_available(self, availability_map: dict[int, int], resource_needed: int, usage_timeline: list[tuple[int, int, int]], after_timestamp: int) -> int:
        """
        Helper function to find the earliest timestamp after a given time when
        a certain amount of a resource becomes available.
        """
        # Start by checking the availability at the 'after_timestamp'
        available_at_start = self._get_available_resource(availability_map.copy(), list(usage_timeline),
                                                          self.total_cpu if availability_map is self.cpu_availability else self.total_memory, after_timestamp)
        if available_at_start >= resource_needed:
            return after_timestamp

        # Consider future availability events
        sorted_events = sorted(list(availability_map.keys()) + [item[1] for item in usage_timeline])
        seen_times = set()

        for event_time in sorted_events:
            if event_time > after_timestamp and event_time not in seen_times:
                seen_times.add(event_time)
                available_at_event = self._get_available_resource(availability_map.copy(), list(usage_timeline),
                                                                  self.total_cpu if availability_map is self.cpu_availability else self.total_memory, event_time)
                if available_at_event >= resource_needed:
                    return event_time

        # If still not found, the resource might never be available in the future
        return float('inf')

    @staticmethod
    def _find_latest_before(sorted_times: list[int], timestamp: int) -> int:
        left, right = 0, len(sorted_times) - 1
        latest_before = 0

        while left <= right:
            mid = int((left + right) / 2)
            if sorted_times[mid] < timestamp:
                latest_before = sorted_times[mid]
                left = mid + 1
            else:
                right = mid - 1
        return latest_before

    def _update_availability(self, availability_map: dict[int, int], timestamp: int, change: int):
        """
        Helper function to update the availability of a resource at a given timestamp.
        """
        if timestamp not in availability_map:
            # Find the latest availability before this timestamp
            sorted_times = sorted(availability_map.keys())
            latest_before = self._find_latest_before(sorted_times, timestamp)
            availability_map[timestamp] = availability_map.get(latest_before, self.total_cpu if availability_map is self.cpu_availability else self.total_memory)
        availability_map[timestamp] += change


class OfflineGraphene:
    def build_schedule(self, graph: nx.DiGraph, m: int, dag_allocation: dict):
        # todo: space in paper takes m for number of machines as an argument
        best = None
        best_completion_time = 0
        for subsets in candidate_troublesome_tasks(graph):
            space = Space(dag_allocation['cpu'], dag_allocation['mem'])
            subset_T, subset_S, subset_P, subset_C = set(subsets[0]), set(subsets[1]), set(subsets[2]), set(subsets[3])
            # if subset_T == {'1/0', '11/0', '11/1', '11/2', '12/0', '12/2', '12/3', '2/0'}:

            # as a modification from the graphene algorithm, since the placing pseudocode only permits placing tasks that have satisfied dependencies, but does not
            # consider how to place troublesome tasks with dependencies first, I have two choices:
            # either place troublesome tasks disregard dependencies or
            # place the troublesome tasks that I can (no dependencies), then
            # carry through leftover troublesome tasks into other subsets.
            # an apparent issue is this may violate their no dead-ends theorem (not fully certain)
            # ie: how can i place a child of a troublesome task if I have never placed it
            # another issue is how do I add it later? theoretically, wouldn't I need to reset the P S and C subsets as they no longer represent
            # children/parents/siblings of all tasks in the troublesome set
            # Since the algorithm doesn't specify, I believe the best way to deal with this is to disregard dependencies
            # I will need to update my placing algorithms to handle placing parents before the troublesome tasks, which I will do later
            space = self.place_tasks(subset_T, space, graph, disregard_dependencies=True)
            # if len(space.placed_tasks()) != len(subset_T):
            #     continue

            # if placing only what I can, include a line like this:
            # subset_S.update(subset_T.difference(set(space.placed_tasks())))

            space = self.try_subset_orders(space, subset_S, subset_P, subset_C, graph)

            space_completion_time = space.completion_time()
            if best is None or space_completion_time < best_completion_time:
                best = space
                best_completion_time = space_completion_time
        return best.task_details(graph)

    def place_tasks(self, subset: set, space: Space, graph: nx.DiGraph, disregard_dependencies: bool = False) -> Space:
        if len(subset) == 0:
            return space
        forward_space = self.place_tasks_forward(subset, space, graph, disregard_dependencies)
        backward_space = self.place_tasks_backward(subset, space, graph, disregard_dependencies)
        if forward_space.completion_time() <= backward_space.completion_time():
            return forward_space
        else:
            return backward_space

    def try_subset_orders(self, space: Space, subset_S: set, subset_P: set, subset_C: set, graph: nx.DiGraph):
        subset_orders = [self.place_tasks_forward(subset_C, self.place_tasks_backward(subset_P, self.place_tasks(subset_S, space, graph, disregard_dependencies=True), graph), graph),  # spc
                         self.place_tasks_backward(subset_P, self.place_tasks_forward(subset_C, self.place_tasks(subset_S, space, graph, disregard_dependencies=True), graph), graph),  # scp
                         self.place_tasks_backward(subset_P, self.place_tasks_backward(subset_S, self.place_tasks_forward(subset_C, space, graph), graph, disregard_dependencies=True), graph),  # csp
                         self.place_tasks_forward(subset_C, self.place_tasks_forward(subset_S, self.place_tasks_backward(subset_P, space, graph), graph, disregard_dependencies=True), graph)]  # psc
        return subset_orders[np.argmin([subset.completion_time() for subset in subset_orders])]

    @staticmethod
    def place_tasks_forward(subset: set, space: Space, graph: nx.DiGraph, disregard_dependencies: bool = False) -> Space:
        space_clone = copy.deepcopy(space)
        finished = set(space_clone.placed_tasks())
        tasks = list(graph.nodes(data=True))
        tasks = list(filter(lambda x: x[0] in subset, tasks))
        while True:
            ready = set()
            longest_task_in_ready = None
            longest_task_details = None
            longest_duration = -1
            longest_task_idx = -1
            for i, (name, details) in enumerate(tasks):
                parents = set(graph.predecessors(name))
                if disregard_dependencies or (len(parents) == 0 or all(p in finished for p in parents)):
                    # no dependencies, can run whenever
                    # or
                    # all dependencies must be done
                    ready.add(name)
                    duration = details['duration']
                    if duration > longest_duration:
                        longest_duration = duration
                        longest_task_in_ready = name
                        longest_task_details = details
                        longest_task_idx = i

            if len(ready) == 0:
                break

            task_dependencies = set(graph.predecessors(longest_task_in_ready)) if disregard_dependencies is False else set()
            if len(task_dependencies) == 0:
                scheduleable_timestamp = 0
            else:
                scheduleable_timestamp = max(space_clone.end_time(task) for task in task_dependencies)

            assert longest_duration > -1
            success = space_clone.insert_task(longest_task_in_ready, longest_task_details['cpu'], longest_task_details['mem'], longest_duration, scheduleable_timestamp)
            finished.add(longest_task_in_ready)
            assert longest_task_idx > -1
            tasks.pop(longest_task_idx)
        return space_clone

    @staticmethod
    def place_tasks_backward(subset: set, space: Space, graph: nx.DiGraph, disregard_dependencies: bool = False):
        """
        This is the exact same as place_tasks_forward but under the assumption that reversing the timeline before and after placing backwards on an initially increasing timeline is
        equivalent to placing backwards properly
        """
        space_clone = copy.copy(space)
        space_clone.invert()
        finished = set(space_clone.placed_tasks())
        tasks = list(graph.nodes(data=True))
        tasks = list(filter(lambda x: x[0] in subset, tasks))
        while True:
            ready = set()
            longest_task_in_ready = None
            longest_task_details = None
            longest_task_idx = -1
            longest_duration = -1
            for i, (name, details) in enumerate(tasks):
                # place backwards, children first, ascending for now
                children = set(graph.successors(name))
                if disregard_dependencies or (len(children) == 0 or all(p in finished for p in children)):
                    # no dependencies, can run whenever
                    # or
                    # all dependencies must be done
                    ready.add(name)
                    duration = details['duration']
                    if duration > longest_duration:
                        longest_duration = duration
                        longest_task_in_ready = name
                        longest_task_details = details
                        longest_task_idx = i

            if len(ready) == 0:
                break

            task_dependencies = set(graph.successors(longest_task_in_ready)) if disregard_dependencies is False else set()
            if len(task_dependencies) == 0:
                scheduleable_timestamp = 0
            else:
                scheduleable_timestamp = max(space_clone.end_time(task) for task in task_dependencies)

            assert longest_duration > -1
            success = space.insert_task(longest_task_in_ready, longest_task_details['cpu'], longest_task_details['mem'], longest_duration, scheduleable_timestamp)
            finished.add(longest_task_in_ready)
            assert longest_task_idx > -1
            tasks.pop(longest_task_idx)
        space_clone.invert()
        return space_clone


def get_batch_instance_column_names():
    names_to_types = {"instance_name": str, "task_name": str, "job_name": str, "task_type": np.int32,
                      "status": str, "start_time": np.int32, "end_time": np.int32,
                      "machine_id": str, "seq_no": np.int32, "total_seq_no": np.int32,
                      "cpu_avg": np.float32, "cpu_max": np.float32, "mem_avg": np.float32, "mem_max": np.float32}

    names = list(names_to_types.keys())
    return names


def get_last_job_timestamp_in_trace():
    """
    Get the latest timestamp a job shows up in the instance trace

    This is used in order to track when a DAG will be considered completely constructed before filtering them for valid DAGs
    """
    trace_file = "../alibaba-trace/batch_instance_sort.csv"
    job_timestamps = {}

    num_lines = 1_228_679_841
    chunksize = 10_000_000
    max_iter = num_lines // chunksize + 1

    early_stop_iter = 0
    max_timestamp = 86400 * 3
    start = time.time()

    os.makedirs("scheduler_data", exist_ok=True)
    output_file = os.path.join("scheduler_data", "last_job_timestamps.csv")
    for j, data in enumerate(pd.read_csv(trace_file, chunksize=chunksize, names=get_batch_instance_column_names())):
        # if j > early_stop_iter:
        #     break
        data.dropna(inplace=True)
        # if data["start_time"].min() > max_timestamp:
        #     break

        for row in data.to_numpy():
            (instance_name, task_name, job_name, task_type, status, start_time, end_time, machine_id, seq_no, total_seq_no, cpu_avg, cpu_max, mem_avg, mem_max) \
                = row
            job_timestamps[job_name] = start_time
        logger.info("time elapsed: %s(s), time left: %s(s), estimated finish time: %s" % calcProcessTime(start, j + 1, max_iter))

    df = pd.DataFrame.from_dict({"job_name": job_timestamps.keys(), "timestamp": job_timestamps.values()})
    df.to_csv(output_file, columns=["job_name", "timestamp"], index=False)


def consume_trace(options: argparse.Namespace):
    """
    Converts batch instance trace into graphs of instances and dumps pickled batches of them into files
    Since there is so much data to parse through, this will use hundreds of gigabytes of memory if left alone.
    It seems like 200GB memory is able to extract aroud 2300 jobs before it crashes due to OOM
    """
    # code taken from my critical_path analysis
    trace_file = "../alibaba-trace/batch_instance_sort.csv"

    alibaba_trace = pd.read_csv("alibaba-2018-sorted.csv")
    alibaba_trace.dropna(inplace=True)
    alibaba_trace = alibaba_trace[["start_time", "end_time", "plan_cpu", "plan_mem", "instance_num", "status", "job_name", "task_name"]]
    # alibaba_dict = {f"{row[6]},{row[7]}": [int(row[1]), int(row[2])] for row in alibaba_trace.to_numpy()}

    # alibaba_dict = {f"{row[0]},{row[1]}": [int(row[2]), int(row[3])] for row in alibaba_trace.to_numpy()}

    # easy but inefficient solution
    # i need to map job to tasks when looking up dependent tasks as I dont know what the dependent task name is (as in, i could look for task 13 but the name is 13_14)
    job_to_tasks = dict()
    alibaba_dict = dict()
    for row in alibaba_trace.to_numpy():
        start_time, end_time, plan_cpu, plan_mem, instance_num, status, job_name, task_name = row
        alibaba_dict[f"{job_name},{task_name}"] = [int(start_time), int(end_time), float(plan_cpu), float(plan_mem), int(instance_num)]
        job_to_tasks.setdefault(job_name, dict())
        # subdict that maps first task number to the actual name, as in if task name is 13_14 then map 13 to 13_14
        if task_name[0] == 't':
            # skip the individual tasks
            continue
        base_task = task_name[1:].split("_")[0]
        job_to_tasks[job_name][base_task] = task_name
    # del alibaba_trace
    last_job_timestamps = os.path.join("scheduler_data", "last_job_timestamps.csv")
    last_job_timestamps_lookup = pd.read_csv(last_job_timestamps).set_index('job_name')['timestamp']

    num_lines = 1_228_679_841
    chunksize = 100_000
    max_iter = num_lines // chunksize + 1
    total = invalid = 0

    max_timestamp = 86400 * 3

    job_dags = {}
    seen_job_instances = {}
    latest_seen_job_timestamp = {}

    good_job_dags = {}

    def task_to_base(task: str) -> str:
        return task.split('_')[0][1:]

    def task_to_dependencies(task: str) -> list[str]:
        return task.split('_')[1:]

    def consider_good_dag(job_name, graph):
        """
        Test if DAG is good. If good, add to set of good dag.
        """
        good_dag = nx.is_directed_acyclic_graph(graph)
        if good_dag:
            good_job_dags[job_name] = graph

    LIMIT = 100
    LIMIT_DUMP_ITERATION = 100
    def dump_good_dags(good_job_dags, dump_iteration):
        # idk why I can't rely on variable scope here
        if len(good_job_dags) > LIMIT:
            dump_filename = os.path.join("scheduler_data", f"job_dags_{dump_iteration}.pkl")
            with open(dump_filename, "wb") as f:
                sliced = dict(islice(good_job_dags.items(), LIMIT))
                pickle.dump(sliced, f)
            for k in sliced.keys():
                del good_job_dags[k]
            logger.info(f"Dumped iteration {dump_iteration}")
            dump_iteration += 1
        if dump_iteration > LIMIT_DUMP_ITERATION:
            return good_job_dags, -1
        return good_job_dags, dump_iteration
    dump_iteration = 0

    def filter_old_dags():
        to_delete = []
        for job_name, dag in job_dags.items():
            # if there are no more instances for this job, try to consider it for simulation
            if latest_seen_job_timestamp[job_name] >= last_job_timestamps_lookup[job_name]:
                consider_good_dag(job_name, dag)
                # get rid of it if good or bad, since either way we wont consider it
                to_delete.append(job_name)
                del latest_seen_job_timestamp[job_name]
        for job_name in to_delete:
            del job_dags[job_name]
    
    start = time.time()
    # limit runtime to 3 hours
    LIMIT_RUNTIME = 60 * 60 * 3
    for j, data in enumerate(pd.read_csv(trace_file, chunksize=chunksize, names=get_batch_instance_column_names())):
        current_time = time.time()
        if current_time - start > LIMIT_RUNTIME:
            break
        data.dropna(inplace=True)

        if data["start_time"].min() > max_timestamp:
            break

        # independent tasks and not dags, ignore these
        data = data[~data["task_name"].str.startswith("task_")]
        # remove Stg tasks
        data = data[~data["task_name"].str.contains("Stg", case=True, na=False)]
        for row in data.to_numpy():
            (instance_name, task_name, job_name, task_type, status, start_time, end_time, machine_id, seq_no, total_seq_no, cpu_avg, cpu_max, mem_avg, mem_max) \
                = row
            job_dags.setdefault(job_name, nx.DiGraph())
            G = job_dags[job_name]

            task_number = task_to_base(task_name)
            requested = alibaba_dict[f"{job_name},{task_name}"]
            plan_cpu = requested[2]
            plan_mem = requested[3]
            # instance name will become an integer from 0 to instance_num, # I suspect there is a better solution
            seen_job_instances.setdefault(job_name, {})
            seen_job_instances[job_name].setdefault(task_name, 0)
            instance_num = seen_job_instances[job_name][task_name]
            seen_job_instances[job_name][task_name] += 1
            current_task_name = f"{task_number}/{instance_num}"
            completion_time = end_time - start_time
            latest_seen_job_timestamp[job_name] = start_time
            # cpu_norm = 4023 * 96
            # mem_norm = 4023
            G.add_node(current_task_name, start_time=start_time, end_time=end_time,
                       # cpu_avg=cpu_avg / cpu_norm, cpu_max=cpu_max / cpu_norm, mem_avg=mem_avg / mem_norm, mem_max=mem_max / mem_norm, plan_cpu=plan_cpu / cpu_norm,
                       # plan_mem=plan_mem / mem_norm,
                       cpu=cpu_avg, mem=mem_avg,
                       job_name=job_name, task_name=task_name,
                       # instance_name=instance_name,
                       duration=completion_time)
            # When adding edges, I am assuming that the production trace does not run a single child instance until all parent instances are done
            dependencies = task_to_dependencies(task_name)
            for parent in dependencies:
                parent_task = job_to_tasks[job_name][parent]
                parent_number = task_to_base(parent_task)
                parent_requested = alibaba_dict[f"{job_name},{parent_task}"]
                parent_instance_num = parent_requested[4]

                for i in range(parent_instance_num):
                    parent_task_name = f"{parent_number}/{i}"
                    # I'm putting the child task completion time, which means each edge is how long until the end of the child
                    G.add_edge(parent_task_name, current_task_name, weight=completion_time)
        filter_old_dags()
        good_job_dags, dump_iteration = dump_good_dags(good_job_dags, dump_iteration)
        if dump_iteration == -1:
            # we surpassed the limit set for amount of dumpfiles to create, so quit out before we pollute the server with terabytes of files
            break


def test_offline():
    with open("scheduler_data/one_job_dag.pkl", "rb") as f:
        test_graph = pickle.load(f)
    offline = OfflineGraphene()
    dag_allocation = {'cpu': 9600, 'mem': 100}
    job_name = list(test_graph.nodes(data=True))[0][1]['job_name']
    task_ordering = offline.build_schedule(test_graph, m=1, dag_allocation=dag_allocation)
    online = OnlineScheduler(4023, 9600, 100, 0.1, 1, 1, 10)
    current_time = 0
    while len(task_ordering) != 0:
        scheduled, task_ordering, next_timestamp = online.schedule_tasks(job_name, task_ordering, current_time)
        current_time = next_timestamp


def get_some():
    with open("scheduler_data/job_dags.pkl", "rb") as f:
        test_graph_dict = pickle.load(f)

    to_delete = []
    keep = 1000
    count = 0
    for k in test_graph_dict.keys():
        if count > keep:
            to_delete.append(k)
        count += 1

    for k in to_delete:
        del test_graph_dict[k]

    with open("scheduler_data/job_dags.pkl", "wb") as f:
        pickle.dump(test_graph_dict, f)


def get_one():
    # with open("scheduler_data/job_dags.pkl", "rb") as f:
    #     test_graph_dict = pickle.load(f)

    # with open("scheduler_data/one_job_dag.pkl", "wb") as f:
    #     another_dict = {}
        # another_dict['j_2583911'] = test_graph_dict['j_2583911']
        # another_dict['j_1288147'] = test_graph_dict['j_1288147']

        # another_dict['j_2227610'] = test_graph_dict['j_2227610']
        # another_dict['j_1719626'] = test_graph_dict['j_1719626']

        # another_dict['j_3795195'] = test_graph_dict['j_3795195']

        # another_dict['j_3727204'] = test_graph_dict['j_3727204']

        # another_dict['j_2635372'] = test_graph_dict['j_2635372']

        # pickle.dump(another_dict, f)
    with open("scheduler_data/offline/offline_job_dags_0.pkl", "rb") as f:
        test_graph_dict = pickle.load(f)

    import pdb; pdb.set_trace()

def fix_offline_dags():
    # since I extract my DAGs from realtime running data, they may not be completely ordered (but almost)

    total_queue = deque()
    all_dependencies = dict()
    for i in range(100):
        path = f"scheduler_data/offline/offline_job_dags_{i}.pkl"
        if not os.path.exists(path):
            logger.info(f"path {path} does not exist")
            break
        logger.info(f"reading path {path}")
        with open(path, "rb") as f:
            to_schedule, task_dependencies = pickle.load(f)
        all_dependencies.update(task_dependencies)

        for item in to_schedule:
            total_queue.append(item)
    total_queue = sorted(total_queue, key=lambda x: x[0])
    i = 0
    while total_queue:
        path = f"scheduler_data/offline/offline_job_dags_{i}.pkl"
        current_queue = deque(islice(total_queue, 0, 100))
        task_dependencies = dict()
        for _, job_name, _ in current_queue:
            task_dependencies[job_name] = all_dependencies[job_name]
        total_queue = deque(islice(total_queue, 100, len(total_queue) + 1))
        i += 1
        logger.info(f"{i}, len {len(total_queue)}")
        with open(path, "wb") as f:
            pickle.dump((current_queue, task_dependencies), f)
                

    
def run_offline_graphene():
    # also get the real schedule time of the jobs
    alibaba_trace = pd.read_csv("alibaba-2018.csv", names=["task_name", "instance_num", "job_name", "task_type", "status", "start_time", "end_time", "plan_cpu", "plan_mem"])
    alibaba_trace = alibaba_trace[["job_name", "start_time"]]
    trace_start_times = alibaba_trace.set_index("job_name")["start_time"]
    trace_start_times = trace_start_times.groupby(["job_name"]).min()
    del alibaba_trace

    for i in range(100):
        # this will run over all available sections
        path=f"scheduler_data/job_dags_{i}.pkl"
        output_path = f"scheduler_data/offline/offline_job_dags_{i}.pkl"
        if not os.path.exists(path):
            break
        with open(path, "rb") as f:
            test_graph_dict = pickle.load(f)

        all_task_orderings = dict()
        bad_jobs = ('j_3805685', 'j_3983852', 'j_2635372')
        # j_2635372 is missing one?? node M2/1 is missing data??
        max_amt = 10000
        i = 0
        max_jobs = 1000
        for job_name, graph in test_graph_dict.items():
            i += 1
            if i > max_amt:
                break
            if job_name in bad_jobs:
                continue

            if len(all_task_orderings) > max_jobs:
                break
            offline = OfflineGraphene()
            dag_allocation = {'cpu': 9600, 'mem': 100}
            job_name = list(graph.nodes(data=True))[0][1]['job_name']

            print(f"found job with {len(list(graph.nodes()))} nodes {job_name}")
            try:
                task_ordering = offline.build_schedule(graph, m=1, dag_allocation=dag_allocation)
            except Exception as e:
                # some jobs seem to be malformed for one reason or another (particularly empty node data in my graph), just ignore them for now
                print(f'{job_name} has exception? {e}')
                continue

            all_task_orderings[job_name] = task_ordering

        task_dependencies = {}
        for job_name, task_infos in all_task_orderings.items():
            task_dependencies.setdefault(job_name, dict())
            for task_info in task_infos:
                task_dependencies[job_name].setdefault(task_info[0], dict())
                task_dependencies[job_name][task_info[0]] = task_info[1][4]

        # consider all jobs at a time in one epoch
        print('offline done')
        # keep the order of the jobs
        # rng = np.random.default_rng(seed=10)
        # random_indices_sequence = rng.choice(
        #     np.arange(0, len(all_task_orderings)),
        #     size=len(all_task_orderings),
        #     replace=False
        # )
        indices_sequence = np.arange(0, len(all_task_orderings))

        to_schedule = deque()
        for i in indices_sequence:
            job_name = list(all_task_orderings.keys())[i]
            e = all_task_orderings[job_name]
            start_time = trace_start_times[job_name]
            if isinstance(start_time, pd.Series):
                start_time = start_time.min()
            to_schedule.append((start_time, job_name, e))

        with open(output_path, "wb") as f:
            pickle.dump((to_schedule, task_dependencies), f)

def retime_offline_jobs(to_schedule, time_inc: int, starting_time: int = 0):
    to_schedule_retimed = deque()
    for _, job_name, e in to_schedule:
        to_schedule_retimed.append((starting_time, job_name, e))
        starting_time += time_inc
    return to_schedule_retimed, starting_time

def sort_alibaba():
    alibaba_trace = pd.read_csv("alibaba-2018.csv", names=["task_name", "instance_num", "job_name", "task_type", "status", "start_time", "end_time", "plan_cpu", "plan_mem"])
    alibaba_trace = alibaba_trace.sort_values('start_time')
    alibaba_trace.dropna(inplace=True)
    alibaba_trace.to_csv("alibaba-2018-sorted.csv", index=False)

def test_stats():
    path = f"scheduler_data/offline/offline_job_dags_0.pkl"
    with open(path, "rb") as f:
        to_schedule, task_dependencies = pickle.load(f)
    alibaba_trace = pd.read_csv("alibaba-2018.csv", names=["task_name", "instance_num", "job_name", "task_type", "status", "start_time", "end_time", "plan_cpu", "plan_mem"])
    alibaba_trace = alibaba_trace[["job_name", "end_time"]]
    trace_end_times = alibaba_trace.set_index("job_name")["end_time"]
    trace_end_times = trace_end_times.groupby(["job_name"]).min()
    del alibaba_trace
    while to_schedule:
        start_time, to_schedule_job_name, to_schedule_task_ordering = to_schedule.popleft()
        logger.info(f"job {to_schedule_job_name} with start time {start_time} has end time {trace_end_times[to_schedule_job_name]}")


def test_multiple_online(options: argparse.Namespace):
    typ = options.typ
    online = OnlineScheduler(1, 9600, 100, 0.1, 1, 1, deficit_threshold=100000)

    scheduleable_jobs = dict()

    def load_job_dags(retime: bool=True):
        starting_time = 0
        to_schedule = deque()
        task_dependencies = dict()
        largest = 0
        # Graphene can be pretty hard to run as optimal scheduling is intractable. While I have 2300 jobs extracted from my offline portion,
        # take only the first chunk of 94
        for i in range(1):
            path = f"scheduler_data/offline/offline_job_dags_{i}.pkl"
            if not os.path.exists(path):
                logger.info(f"path {path} does not exist")
                break
            with open(path, "rb") as f:
                new_to_schedule, new_task_dependencies = pickle.load(f)
            
            best_new_to_schedule, best_new_task_dependencies = None, None

            if True:
                # similarly to the intractible reason above, take only the first n jobs. Else graphene can run for days.
                to_schedule += deque(islice(new_to_schedule, 0, 20))
                task_dependencies.update(new_task_dependencies)
            else:
                # Create a synthetic trace consisting only of the largest job seen so far
                for schedule in new_to_schedule:
                    num_dependencies = len(new_to_schedule[2])
                    if num_dependencies > largest:
                        best_new_to_schedule = schedule
                        best_new_task_dependencies = new_task_dependencies[best_new_to_schedule[1]]

                for i in range(len(new_to_schedule)):
                    schedule = best_new_to_schedule
                    dependencies = best_new_task_dependencies
                    new_job_name = f'j_{np.random.randint(20000,100000)}'
                    to_schedule.append((schedule[0], new_job_name, schedule[2]))
                    task_dependencies.update({new_job_name: dependencies})
            if retime:
                to_schedule, starting_time = retime_offline_jobs(to_schedule, 5, starting_time)
        
            logger.info(f"reading path {path} with {len(to_schedule)} jobs")
        return to_schedule, task_dependencies

    to_schedule, task_dependencies = load_job_dags(retime=True)

    # I forgot to preprocess for duplicates in the job dags, remove duplicates here
    def remove_duplicates(schedule: deque):
        new_schedule = deque()
        seen = set()
        for item in schedule:
            if item[1] not in seen:
                seen.add(item[1])
                new_schedule.append(item)
        return new_schedule
    to_schedule = remove_duplicates(to_schedule)
    for job_name, task_dep_info in task_dependencies.items():
        for task_name, task_dep in task_dep_info.items():
            online.add_predecessors_for_job(job_name, task_name, task_dep)

    current_time = to_schedule[0][0]
    while True:
        if len(to_schedule) == 0 and len(scheduleable_jobs) == 0:
            break
        if len(to_schedule) > 0:
            online.next_timestamps.add(to_schedule[0][0])
            if current_time >= to_schedule[0][0]:
                _, to_schedule_job_name, to_schedule_task_ordering = to_schedule.popleft()
                scheduleable_jobs[to_schedule_job_name] = to_schedule_task_ordering
        if len(scheduleable_jobs) == 0:
            # this is some crappy workaround that I have to do since I just realized a bug when there are no tasks to schedule but running tasks can end
            # this doesnt actually affect the scheduler in any way but affects the logging
            online.check_finished(current_time)
            online.log_usage(current_time)
        old_time = current_time
        scheduleable_jobs, current_time = online.schedule_tasks_bunch(scheduleable_jobs, current_time, typ)
        logger.info(f"cluster usage {online.timestamp_to_usage[old_time]} as timestamp {current_time}")

    logger.info('finishing')
    online.finish_rest(current_time)

    output_path = f"{typ.lower()}_utilization.png"
    online.plot_cluster_usage(output_path)
    save_path = f"{typ.lower()}_utilization_smallest.csv"
    online.dump_cluster_usage(save_path)

def test_online_graph(options: argparse.Namespace):
    plt.rcParams.update({
        "font.family": 'TeX Gyre Schola Math'
    })
    typ = options.typ
    online = OnlineScheduler(1, 9600, 100, 0.1, 1, 1, deficit_threshold=100000)
    save_path = f"{typ.lower()}_utilization_large_jobs.csv"
    output_path = f"{typ.lower()}_utilization_large_jobs_from_csv.png"
    online.plot_cluster_usage(output_path, save_path, sigma=3)
    save_path = f"{typ.lower()}_utilization_small.csv"
    output_path = f"{typ.lower()}_utilization_small_from_csv.png"
    online.plot_cluster_usage(output_path, save_path, sigma=3)
    save_path = f"{typ.lower()}_utilization_smallest.csv"
    output_path = f"{typ.lower()}_utilization_smallest_from_csv.png"
    online.plot_cluster_usage(output_path, save_path, sigma=3)
    


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--operation", "-o", default=None)
    parser.add_argument("--skip-amt", type=int, default=None)
    parser.add_argument("--typ", "-t", default="graphene")
    options = parser.parse_args(args)

    output_file = os.path.join("scheduler_data", "last_job_timestamps.csv")
    if not os.path.exists(output_file) and options.skip is False:
        get_last_job_timestamp_in_trace()

    os.makedirs("scheduler_data/offline", exist_ok=True)

    # consume_trace()
    if options.operation == "get_one":
        get_one()
    elif options.operation == "get_some":
        get_some()
    elif options.operation == "consume":
        consume_trace(options)
    elif options.operation == "offline_graphene":
        run_offline_graphene()
    elif options.operation == "multiple_online":
        test_multiple_online(options)
    elif options.operation == "fix_offline":
        fix_offline_dags()
    elif options.operation == "test":
        test_stats()
    elif options.operation == "sort_alibaba":
        sort_alibaba()
    elif options.operation == "graph":
        test_online_graph(options)
    else:
        test_offline()


if __name__ == "__main__":
    main()

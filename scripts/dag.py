from __future__ import annotations

import logging
import re
import argparse
import sys
from typing import Tuple, List, Union, Dict, TypedDict, Any, Optional, Set, cast

import coloredlogs

from scheduler_job import Job

# number_regex = r"\d+"
# number_regex_c = re.compile(number_regex)

task_id_regex = r"[._]"
task_id_regex_c = re.compile(task_id_regex)

# takes O(n) memory which is probably fine
completed_tasks: Set[Tuple[str, str]] = set()

logger = logging.getLogger(__name__)


class TaskDAG(Job):
    def __init__(self, task_id: str, task_dependencies: List[str], job_name: str,
                 wid: str, arrival_time: str, end_time: str, cpu: str, rss: str,
                 instance_num: str = "1", task_name: Optional[str] = None, lifetime: Optional[str] = None, **kwargs) -> None:
        # instance num currently doesn't represent anything
        # no dependencies, is an independent task (may have children)
        self.independent = len(task_dependencies) == 0
        self.id = task_id
        self.dependencies = task_dependencies

        # tasks that are not in a DAG will have a random string, so detect this
        self.is_dag = self.id.isdigit()
        # task names are only unique per job
        # self.job and self.job_name are duplicated
        self.job = job_name

        # for bookkeeping in sampling
        self._lifetime = lifetime
        super().__init__(wid, arrival_time, end_time, cpu, rss, job_name, task_name, instances=int(instance_num))

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def __str__(self) -> str:
        return f"{self.job}, {self.id}"

    def set_done(self) -> None:
        # it seems like task names are not job independent
        # completed_tasks.add((self.id, self.job))
        completed_tasks.add((self.job, self.id))

    def dependencies_done(self) -> bool:
        if self.independent:
            return True
        for d in self.dependencies:
            if not lookup_task(self.job, d):
                return False
        return True

    @classmethod
    def job_to_taskdag(cls, job: Job) -> "TaskDAG":
        assert job.job_name is not None
        assert job.task_name is not None
        task = generate_task(job)
        copy_job_metadata_to_task(job, task)
        return task


class TaskInstance(TaskDAG):
    def __init__(self, wid: int, **kwargs) -> None:
        self.wid = wid
        super().__init__(wid=wid, **kwargs)

    @classmethod
    def task_to_taskinstance(cls, wid: int, task: TaskDAG) -> TaskInstance:
        return TaskInstance(wid=wid, task_id=task.task_name, task_dependencies=[], job_name=task.job_name, instance_num=task.instances,
                            arrival_time=task.arrival_time, end_time=task._end_time, cpu=task.cpu, rss=task.rss)


class JobDAG:
    def __init__(self, job_name: str, tasks: List[TaskDAG]) -> None:
        self.id = job_name
        root_node = list()
        # find the head nodes by grabbing the only node with a 0 in-degree
        # for task in tasks:
        #     if len(task.dependencies) == 0:
        #         root_node.append(task)

        # print([task.task_name for task in tasks], [task.task_name for task in root_node])
        # if len(root_node) == 0:
        #     logger.debug("uh oh")
        #     raise RuntimeError("DAG is not acyclical!")
        # note: this does not literally create a dag, simply holds all relevant tasks
        self.tasks = tasks
        self.task_ids = {task.id for task in tasks}
        self.job_name = job_name
        # self.tasks_completed: Dict[str, bool] = {task.id: False for task in tasks}

    def add_dependency(self, new_tasks: List[TaskDAG]) -> None:
        # for task in new_tasks:
        # assumes the task is new
        # self.tasks_completed[task.id] = False
        self.tasks.extend(new_tasks)
        for task in new_tasks:
            self.task_ids.add(task.id)

    def valid_graph(self) -> bool:
        """
        Some of the tasks are missing in the trace resulting in broken graphs
        Test and return false if the graph is invalid
        """
        for task in self.tasks:
            for d in task.dependencies:
                # ensure the dependency exists
                if d not in self.task_ids:
                    # does not exist, is invalid
                    logger.info(f"Job DAG {self.tasks} is invalid; {task} nonexistent dependency {d}")
                    return False
        return True

    def __hash__(self):
        return hash(self.id)

    def task_exists(self, task: TaskDAG) -> bool:
        if task.job_name != self.id:
            return False
        if task.id in self.task_ids:
            return True
        # print(f"job match {task.job_name} but does not have task {task.id}")
        return False


def lookup_task(job_id: str, task_id: str) -> bool:
    return (job_id, task_id) in completed_tasks


def parse_task_name(task_name: str) -> Tuple[str, List[str]]:
    """
    Parse a task name to get the dependencies assuming the task name is defined Alibaba style

    Returns the base task ID and a list of the task IDs the base task depends on

    There are two types of task names, one where the task name is a random string and one where
    it specifies a dependency, so the return type will only be a string but may not be int coercible
    """
    # some tasks are not DAGs, which are named with random characters
    # the examples have them ending in two equal signs, so for now, check against that
    # also check if the name starts with task to be safe
    if task_name.endswith("==") or task_name.startswith("task"):
        return task_name, []
    # this regex doesn't work because sometimes the names are not numerical
    # task_ids = number_regex_c.findall(task_name)
    # I parsed through the trace and it is guaranteed that task_names start with R, M, J, L, or task, so get rid of the first letter
    # it also seems like task names are not job independent
    task_ids = task_id_regex_c.split(task_name[1:])
    if len(task_ids) == 1:
        # this task is independent
        return task_ids[0], []
    # The way alibaba structures dependencies in strings, is the base task is first and everything
    # after is a task the base depends on
    return task_ids[0], task_ids[1:]


def parse_task_name_with_prefix(task_name: str) -> Tuple[Optional[str], str, List[str]]:
    """
    Parse a task name to get the dependencies assuming the task name is defined Alibaba style

    Returns the base task ID and a list of the task IDs the base task depends on

    There are two types of task names, one where the task name is a random string and one where
    it specifies a dependency, so the return type will only be a string but may not be int coercible
    """
    # some tasks are not DAGs, which are named with random characters
    # the examples have them ending in two equal signs, so for now, check against that
    # also check if the name starts with task to be safe
    if task_name.endswith("==") or task_name.startswith("task"):
        return None, task_name, []
    # this regex doesn't work because sometimes the names are not numerical
    # task_ids = number_regex_c.findall(task_name)
    # I parsed through the trace and it is guaranteed that task_names start with R, M, J, L, or task, so get rid of the first letter
    # it also seems like task names are not job independent
    task_ids = task_id_regex_c.split(task_name[1:])
    if len(task_ids) == 1:
        # this task is independent
        return task_name[0], task_ids[0], []
    # The way alibaba structures dependencies in strings, is the base task is first and everything
    # after is a task the base depends on
    return task_name[0], task_ids[0], task_ids[1:]


def copy_job_metadata_to_task(job: Job, task: TaskDAG) -> None:
    # everything in job.__init__ but copied over to task
    # using the init is fine unless the job has been changed since, which is likely
    task.wid = job.wid
    task.arrival_time = job.arrival_time
    task.exec_time = job.exec_time
    task.cpu = job.cpu
    task.rss = job.rss
    task.finish_time = job.finish_time
    task.start_time = job.start_time
    task.state = job.state  # Initial state
    task.state_start_times = job.state_start_times  # Track when each state starts
    task.state_durations = job.state_durations  # Track duration of each state
    task.location = job.location
    task.job_name = job.job_name
    task.task_name = job.task_name
    task.instances = job.instances


def generate_dummy_task(task_name: str, job_name: str, instance_num: str) -> TaskDAG:
    """
    Instantiates a dummy task dag with no config information, holds no relevant information of the parent class fields
    """
    base_task_id, task_dependencies = parse_task_name(task_name)
    return TaskDAG(task_id=base_task_id, job_name=job_name, instance_num=instance_num, task_name=task_name, task_dependencies=task_dependencies,
                   wid='0', arrival_time='0', end_time='0', cpu='0', rss='0')


def generate_task(job: Job) -> TaskDAG:
    return generate_task_args(job.task_name, job.job_name, instance_num=job.instances,
                              wid=job.wid, arrival_time=job.arrival_time, end_time=job._end_time, cpu=job.cpu, rss=job.rss)


def generate_task_args(task_name: str, job_name: str, instance_num: str,
                       wid, arrival_time, end_time, cpu, rss, lifetime=None) -> TaskDAG:
    """
    Given a task name, create a task out of it, filling out dependencies as necessary
    """
    base_task_id, task_dependencies = parse_task_name(task_name)
    return TaskDAG(base_task_id, task_dependencies, instance_num=instance_num, job_name=job_name, wid=wid, arrival_time=arrival_time,
                   end_time=end_time, cpu=cpu, rss=rss, task_name=task_name, lifetime=lifetime)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')

    options = parser.parse_args(args)
    output = parse_task_name(options.input)
    print(output)


if __name__ == "__main__":
    main()

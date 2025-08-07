import argparse
import sys

import pandas as pd

from dag import JobDAG, generate_task_args


def validate_dag(trace_file: str) -> bool:
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
    # See if the trace contains an invalid dag
    current_count = 0
    for job_name, job in jobs.items():
        current_count += 1
        if not job.valid_graph():
            return False
    return True


def validate_all() -> bool:
    # hard coded paths
    probability = "0.001"
    for i in range(1, 11):
        file = f"Alibaba/trace-sample/dag_updated_vms_{probability}_{i}.csv"
        if not validate_dag(file):
            return False

    probability = "0.0001"
    for i in range(1, 11):
        file = f"Alibaba/trace-sample/dag_updated_vms_{probability}_{i}.csv"
        if not validate_dag(file):
            return False

    probability = "0.00001"
    for i in range(1, 101):
        file = f"Alibaba/trace-sample/dag_updated_vms_{probability}_{i}.csv"
        if not validate_dag(file):
            return False
    return True


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--trace-file", "-t")
    # options = parser.parse_args(args)
    valid = validate_all()
    if not valid:
        print("INVALID TRACE")
        sys.exit(1)
    print("ALL TRACES VALID")


if __name__ == "__main__":
    main()

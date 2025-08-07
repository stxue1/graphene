"""
Script to automate running giant set of sample traces with management for queueing runs over a long period of time.
Ideal for giant sets of samples and very long runtimes.

To use this, specify the base sample path, policy, and max CPU usage of the machine. Also specify the range (ie 0-50) and step size. The step size is the amount of
processes that will be scheduled to run at a time, never exceeding the max CPU usage

"""
from importlib import import_module
import sys
from pathlib import Path, PurePath
import os
import argparse
import textwrap
import subprocess
from time import sleep

sample_module = import_module("sample-run")
run_main = getattr(sample_module, "main")

poll_cpu_command = textwrap.dedent("""top -bn1 | grep "Cpu(s)" | \\
       sed "s/.*, *\\([0-9.]*\\)%* id.*/\\1/" | \\
       awk '{print 100 - $1"%"}'""")


def check_cpu_usage() -> float:
    p = subprocess.run(poll_cpu_command, shell=True, stdout=subprocess.PIPE)

    percentage_str = p.stdout.decode("utf-8")

    percentage = float(percentage_str.split("%")[0])
    return percentage


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sample-path", "-d", "-p", dest="trace", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--max-usage", "-m", dest="max", type=float, required=True)
    parser.add_argument("--range", "-r", required=True)
    parser.add_argument("--step", "-s", required=True, type=int)

    options = parser.parse_args(args)

    assert isinstance(options.range, str)
    start, end = options.range.split("-")
    start = int(start)
    end = int(end)
    step = options.step

    curr_start = start
    curr_end = curr_start + step - 1
    if curr_end > end:
        curr_end = end
    last = False
    while True:
        percentage_env = os.environ.get("PERCENTAGE_OVERRIDE")
        try:
            if percentage_env is not None:
                percentage_max = float(percentage_env)
            else:
                percentage_max = options.max
        except ValueError:
            percentage_max = options.max
        percentage = check_cpu_usage()
        if percentage > percentage_max:
            print(f"CPU usage at {percentage}... Waiting 10 seconds...")
            sleep(10)
            continue
        else:
            command = ["--base-sample-path", options.trace, "--policy", options.policy, "--range", f"{curr_start}-{curr_end}", "--max-processes", "50"]
            print(f"Running {command}")
            run_main(command)
            print("Waiting 10 seconds...")
            sleep(10)
        if curr_end >= end:
            return
        curr_start = curr_end + 1
        curr_end = curr_end + step
        if curr_end > end:
            curr_end = end
        if curr_start > end:
            curr_start = end


if __name__ == "__main__":
    main()

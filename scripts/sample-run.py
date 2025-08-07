"""
Run the 2-run.py script in parallel.

Ideal for running on multiple sample traces. To run this script,
specify the base sample path, the sample count, and the policy, and the max processes to run (which represents CPU usage of a machine)
Specify the range to specify a range of sample files, for example, if samples0-50 exists, specify --range 0-20 to run samples 0-20

All threads/scripts will be ran at once.
"""
from importlib import import_module
import sys
from pathlib import Path, PurePath
import os
import argparse

run_import = import_module("2-run")
run_main = getattr(run_import, "main")

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sample-path", "-d", "-p", dest="directory", required=True)
    parser.add_argument("--sample-count", "-s", dest="samples", type=int)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--max-processes", "-m", dest="max", type=int, required=True)
    parser.add_argument("--range")

    options = parser.parse_args(args)

    sample_paths = list()

    base_directory_path = PurePath(options.directory)
    
    basename = base_directory_path.name
    parent = base_directory_path.parent

    idx = basename.rfind("_")
    filename_ext = os.path.splitext(basename)[1]
    start_filename = basename[:idx]
    # sample_num = int(basename[idx+1:])

    if options.range is None:
        for i in range(1, 1+options.samples):
            csv_file = Path(parent) / Path(start_filename + "_" + str(i) + filename_ext)
            sample_paths.append(csv_file)
            print(csv_file)
    else:
        assert isinstance(options.range, str)
        start, end = options.range.split("-")
        start = int(start)
        end = int(end)
        for i in range(start, end + 1):
            csv_file = Path(parent) / Path(start_filename + "_" + str(i) + filename_ext)
            sample_paths.append(csv_file)
            print(csv_file)

    if len(sample_paths) * 4 > options.max:
        raise RuntimeError(f"Tried to run {str(len(sample_paths) * 4)} processes when the max is {str(options.max)}!")
    for sample_path in sample_paths:
        command = ["--use-sample-path", str(sample_path), "--policy", options.policy]
        run_main(command)


if __name__ == "__main__":
    main()
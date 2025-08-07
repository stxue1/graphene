import argparse
import csv
import os
import sys
from pathlib import Path
from typing import TypedDict, List

import numpy as np
import pandas as pd
import logging

from matplotlib import pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_trace")
logger.setLevel(logging.INFO)

Job_Lengths = TypedDict("Job_Lengths", {"name": str, "start": int, "end": int})

# classifications:
# We examine a range of job lengths that map to interactive jobs (1 minute or less), small batch jobs (1hr to 24hrs),
# long batch jobs (24-168hrs), and uninterruptible service jobs (>168hrs).
# The range of job lengths and values within that range are based on version 3 of Google’s Borg cluster trace [7, 47].

# Alibaba's units are in seconds according to https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md
minute = 60
hour = 60 * minute
job_lengths: List[Job_Lengths] = [{"name": "Less than Minute", "start": 0, "end": minute},
                                  {"name": "B/W Minute and Hour", "start": minute, "end": hour},  # has a gap in categorization
                                  {"name": "B/W Hour and Day", "start": hour, "end": 24 * hour},
                                  {"name": "B/W Day and Week", "start": 24 * hour, "end": 168 * hour},
                                  # {"name": "Longer than Week", "start": 168 * hour, "end": sys.maxsize}
                                  ]

save_file = os.path.join("figures", "classify_alibaba_dag.csv")

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", "-t", help="The trace file to analyze")
    parser.add_argument("--directory", "-d", help="The directory containing the source trace file to analyze")
    parser.add_argument("--dont-show", action="store_true", default=False)

    options = parser.parse_args(args)
    if not os.path.exists(save_file):
        do_collect(options)
    do_plot()

def do_collect(options):
    trace_file = None
    if options.trace_file is not None and options.directory is not None:
        raise RuntimeError(f"Only one of --trace-file or --directory can be set!")
    if options.trace_file is not None:
        trace_file = options.trace_file
    else:
        assert options.directory is not None
        trace_file_parent = Path(options.directory) / "trace"
        found = False
        for file_name in os.listdir(trace_file_parent):
            if file_name.endswith('.csv'):
                trace_file = os.path.join(trace_file_parent, file_name)
                found = True
                break
        if not found:
            raise RuntimeError(f"Trace file not found in directory {trace_file_parent}")


    job_class_count = {job_length["name"]: 0 for job_length in job_lengths}
    assert trace_file is not None
    with open(trace_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for id, row in enumerate(reader, 1):  # Use enumerate to get the row index (1-based)
            # Start Time,End Time,Core,Memory,Instance Num,Lifetime  => alibaba
            vmcreated, vmdeleted, vmcorecount, RSS, instances, _, _, _= row
            job_length = int(vmdeleted) - int(vmcreated)
            # get classification in order where range is (exclusive, inclusive)
            classified = False
            for possible in job_lengths:
                # job_length > possible["start"]  # I believe this should always be true if iterating in order
                if job_length <= possible["end"]:
                    class_name = possible["name"]
                    job_class_count[class_name] += int(float(instances))
                    classified = True
                    break
            if not classified:
                print(f"Warning, job that lasted {job_length} seconds long escaped classification")

    labels = np.array([job_length["name"] for job_length in job_lengths])
    sizes = np.array([job_class_count[key] for key in labels])
    df = pd.DataFrame({"sizes": sizes})
    df.to_csv(save_file, index=False)
    print(job_class_count)

def do_plot():
    df = pd.read_csv(save_file)
    sizes = df["sizes"]
    labels = np.array([job_length["name"] for job_length in job_lengths])
    percentages = 100.*sizes/sizes.sum()
    # changes labels to also show percentages
    labels = [f"{label} - {percent:.4f}%" for label, percent in zip(labels, percentages)]
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis", len(labels) + 1)
    colors = [cmap(i) for i in range(len(labels) + 1)]
    patches, text = ax.pie(sizes, colors=colors, startangle=90, radius=1.2)
    ax.legend(patches, labels, loc="lower left", bbox_to_anchor=(-0.1, 1.), fontsize=12)
    ax.axis("equal")
    plot_filename = "alibaba_job_lengths.pdf"
    plt.savefig(plot_filename, bbox_inches="tight")
    print(f"Saved plot to {plot_filename}")

    plt.close()



if __name__ == "__main__":
    main()

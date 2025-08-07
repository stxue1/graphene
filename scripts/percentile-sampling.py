import argparse
import os
import sys
from decimal import Decimal
from typing import Optional

import pandas as pd
import numpy as np
from timeit import default_timer as timer

def percentile_sampling(vms: pd.DataFrame, percentile: Decimal, field: str):
    if field == "core":
        p = vms.Core.quantile(percentile)
        return vms[vms.Core > p]
    elif field == "lifetime":
        p = vms.Lifetime.quantile(percentile)
        return vms[vms.Lifetime > p]
    elif field == "memory":
        p = vms.Memory.quantile(percentile)
        return vms[vms.Memory > p]
    else:
        raise NotImplementedError("Dataframe percentile sampling for field %s is not implemented." % field)

def process_directory(input_directory, field: str, percentile: Decimal = Decimal(0.9), samples_count=1, trace_file_override: Optional[str] = None) -> None:
    # Create a new directory to store sampled CSV files
    output_directory = os.path.join(input_directory, 'trace-sample')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all files in the input directory
    trace_file_parent = os.path.join(input_directory, 'trace') if trace_file_override is None else os.path.dirname(trace_file_override)
    start = timer()
    is_alibaba = input_directory == "Alibaba"
    for file_name in os.listdir(trace_file_parent):
        if file_name.endswith('.csv'):
            # Load the CSV file
            file_path = os.path.join(input_directory, 'trace', file_name)
            vms = pd.read_csv(file_path)

            # Alibaba is represented a little differently; each entry has a number of instances launched, for uniform sampling,
            # expand those collapsed entries to one entry per instance
            if is_alibaba:
                v = vms.values
                # from my testing, this needs 59.4 GiB of memory, 800 seconds on server
                vms = pd.DataFrame(v.repeat(vms["Instance Num"].to_numpy(dtype=int), axis=0), columns=vms.columns)

            # Generate specified number of samples and save them
            for i in range(1, samples_count + 1):
                sampled_vms = percentile_sampling(vms, percentile, field)
                # need to recollapse to avoid large file sizes

                sample_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_percentile_{percentile}_{field}_{i}.csv")

                sampled_vms.to_csv(sample_file_path, index=False)
                print(f"Sample {i} saved for {file_name} at {sample_file_path}")
    end = timer()
    print(f"Processed sampling in {end - start} seconds")


def main(args=None):
    # Example usage
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Directory containing the trace files", default=None)
    parser.add_argument("-p", "--percentile", help="Percentile for when running percentile sampling over the trace files.",
                        default=0.08, type=Decimal)
    parser.add_argument("--field", "-f", help="Field in the dataframe to use for the percentile sampling.", choices=["core", "lifetime", "memory"], required=True)
    parser.add_argument("-s", "--samples", help="Number of samples to produce.", default=1, type=int)
    parser.add_argument("--trace-file", help="Point to the trace file to read from. If nonexistent, will assume it is under the supplied --directory.",
                        default=None)
    options = parser.parse_args(args)
    input_directory = options.directory or 'GoogleA/'
    process_directory(input_directory, field=options.field, percentile=options.percentile, samples_count=options.samples, trace_file_override=options.trace_file)


if __name__ == "__main__":
    main()

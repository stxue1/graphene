import argparse
import math
import os
import sys
from decimal import Decimal
from typing import Optional, List, Dict
import pandas as pd
from timeit import default_timer as timer
import logging

logger = logging.getLogger(__name__)

import numpy as np

weights: Dict[int, List[Decimal]] = {}


# qps in jaeger stands for queries per second


def get_weights(length: int) -> List[Decimal]:
    # based off of jaeger's implementation
    # https://github.com/jaegertracing/jaeger/blob/9d709579e4cb1b29bf35e0c516a4e7f0d195bc05/plugin/sampling/strategyprovider/adaptive/weightvectorcache.go#L28-L47
    # essentially w(i)=i^4
    global weights
    if weights.get(length) is not None:
        return weights[length]
    calc_weights = list()
    sum_weights = 0
    for i in range(length, 0, -1):
        w = int(math.pow(i, 4))
        calc_weights.append(Decimal(w))
        sum_weights += w
    for i in range(length):
        calc_weights[i] /= sum_weights
    weights[length] = calc_weights
    return calc_weights


def calculate_weighted(rates: List[Decimal]) -> Decimal:
    """
    Most recent should be at head of list
    """
    # Calculates a weighted average
    # https://github.com/jaegertracing/jaeger/blob/9d709579e4cb1b29bf35e0c516a4e7f0d195bc05/plugin/sampling/strategyprovider/adaptive/processor.go#L343-L353
    if len(rates) == 0:
        return Decimal(0)
    curr: Decimal = Decimal(0)
    w = get_weights(len(rates))
    for i in range(len(rates)):
        curr += rates[i] * w[i]
    return curr


def calculate_new_probability(target_rate: Decimal, rate: Decimal, current_probability: Decimal, percent_increase_cap: Decimal) -> Decimal:
    factor = target_rate / rate
    new_probability = current_probability * factor
    # if rate is lower than target, increase probability slowly
    # if rate is lower, jump directly
    if factor > 1:
        percent_increase = (new_probability - current_probability) / current_probability
        if percent_increase > percent_increase_cap:
            new_probability = current_probability + (current_probability * percent_increase_cap)
    return new_probability


def calculate_rate(count: int, interval: Decimal) -> Decimal:
    """
    :param count: number of scheduled vms
    :param interval: interval of time (seconds)
    :return:
    """
    return count / interval


def within_tolerance(actual: Decimal, expected: Decimal, delta_tolerance: Decimal) -> bool:
    return abs(actual - expected) / expected < delta_tolerance


def decimal_equals(one: Decimal, two: Decimal) -> bool:
    return abs(one - two) < sys.float_info.epsilon  # with Decimal this may not be needed


def calculate_probability(current_probability: Optional[Decimal], target_rate: Decimal, rate: Decimal, initial_sampling_probability: Decimal,
                          delta_tolerance: Decimal, percent_increase_cap: Decimal) -> Decimal:
    # Remember to update previous probability before this call
    # current_probability of None means the start of the algorithm, so set to base
    # https://github.com/jaegertracing/jaeger/blob/9d709579e4cb1b29bf35e0c516a4e7f0d195bc05/plugin/sampling/strategyprovider/adaptive/processor.go#L386-L418
    if current_probability is None:  # starting
        current_probability = initial_sampling_probability

    if within_tolerance(rate, target_rate, delta_tolerance):
        return current_probability

    if decimal_equals(rate, Decimal(0)):
        # Edge case, force sampling of at least one span
        new_probability = current_probability * Decimal(2)
    else:
        new_probability = calculate_new_probability(target_rate, rate, current_probability, percent_increase_cap)
    # jaeger defaults
    max_sampling_probability = Decimal(1)
    min_sampling_probability = Decimal(10 ** (-5))  # one every 100k
    return Decimal.min(max_sampling_probability, Decimal.max(min_sampling_probability, new_probability))


def sampling_alg(vms: pd.DataFrame, options: argparse.Namespace):
    initial_sampling_probability: Decimal = options.initial_sampling_probability
    target_rate: Decimal = options.target_rate
    sample_interval: Decimal = Decimal(options.sample_interval)  # seconds
    delta_tolerance: Decimal = options.delta_tolerance
    percent_increase_cap: Decimal = options.percent_increase_cap
    unit = options.timestamp_unit
    if unit == "microseconds":
        sample_interval_units = sample_interval * 1000000
    elif unit == "milliseconds":
        sample_interval_units = sample_interval * 1000
    else:
        sample_interval_units = sample_interval
        pass

    new_vms = dict()
    vms_np = vms.to_numpy()

    last_interval_start = None
    interval_count = 0

    previous_intervals = list()

    probability = None
    rng = np.random.default_rng()

    # 1000 samples or 1000 seconds
    i = 0
    for (data_idx, data) in enumerate(vms_np):
        start = timer()
        start_time = data[0]
        # end_time = data[1]
        # core = data[2]
        # memory = data[3]

        # if start_time != 0:
        #     print("helo")
        # else:
        #     print(start_time, count)
        #     count += 1

        if last_interval_start is None:
            last_interval_start = start_time

        if start_time >= last_interval_start + sample_interval_units:
            # next interval

            curr_rate = calculate_rate(interval_count, sample_interval)  # rate at current interval
            previous_intervals.append(curr_rate)  # running list of rates for each interval including current
            if len(previous_intervals) > 1000:
                previous_intervals = previous_intervals[-1000:]

            weighted_rate = calculate_weighted(previous_intervals)
            probability = calculate_probability(probability, target_rate, weighted_rate, initial_sampling_probability, delta_tolerance, percent_increase_cap)

            # I *think* probability is per entry (i can't find the exact line of code that references this)
            # Dictionaries are faster for pandas and Python 3.7+ have ordered dictionaries
            for j in range(interval_count):
                if probability > rng.random():
                    new_vms[i] = vms_np[data_idx - interval_count + j]
                    i += 1

            interval_count = 0
            last_interval_start = start_time

        interval_count += 1
        end = timer()

        logger.info(f"took {end - start} seconds for {i}")

    new_vms_df = pd.DataFrame.from_dict(new_vms, "index", columns=vms.columns)
    return new_vms_df


def process_directory(options: argparse.Namespace) -> None:
    # Create a new directory to store sampled CSV files

    input_directory = options.directory
    samples_count = options.samples
    trace_file_override = options.trace_file
    output_directory = os.path.join(input_directory, 'trace-sample')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all files in the input directory
    trace_file_parent = os.path.join(input_directory, 'trace') if trace_file_override is None else os.path.dirname(trace_file_override)
    for file_name in os.listdir(trace_file_parent):
        # Load the CSV file
        file_path = os.path.join(input_directory, 'trace', file_name)
        vms = pd.read_csv(file_path)

        # Generate specified number of samples and save them
        for i in range(1, samples_count + 1):
            sampled_vms = sampling_alg(vms, options)
            sample_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_adaptive_{options.target_rate}rps_{i}.csv")

            sampled_vms.to_csv(sample_file_path, index=False)
            logger.info(f"Sample {i} saved for {file_name} at {sample_file_path}")


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--directory", help="Directory containing the trace files", default=None)
    parser.add_argument("-s", "--samples", help="Number of samples to produce.", default=1, type=int)
    parser.add_argument("--trace-file", help="Point to the trace file to read from. If nonexistent, will assume it is under the supplied --directory.",
                        default=None)
    parser.add_argument("--initial-sampling-probability", "--initial", "-i", help="Initial sampling probability to start with.",
                        dest="initial_sampling_probability", type=Decimal, default=Decimal(1) / Decimal(1000))
    parser.add_argument("--target-rate", "-t", help="The target rate to hit. Unit is number of requests per second.", type=Decimal, default=Decimal(2), dest="target_rate")
    parser.add_argument("--sample-interval", "--interval", "--span", help="How long a span/interval should be in seconds.", type=int, default=1000, dest="sample_interval")
    parser.add_argument("--delta-tolerance", dest="delta_tolerance", type=Decimal, help="Threshold to consider two rates equal.", default=Decimal(3) / Decimal(10))
    parser.add_argument("--percent-increase-cap", "--increase-cap", dest="percent_increase_cap", help="Maximum amount to increase the probability by per each span.",
                        type=Decimal, default=Decimal(1) / Decimal(2))
    # https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md Azure seems to be seconds (timestamp every 5 minutes)
    parser.add_argument("--timestamp-unit", help="Specify the unit of time the original trace file's timestamps are in. (Google is microseconds, Azure is seconds...)",
                        choices=["microseconds", "milliseconds", "seconds"])
    # log_file = 'adaptive_sampling.log'
    # if os.path.exists(log_file):
    #     os.remove(log_file)
    # logging.basicConfig(filename=log_file, level=logging.DEBUG)
    # logger.setLevel(logging.DEBUG)
    options = parser.parse_args(args)
    process_directory(options)


if __name__ == "__main__":
    main()

import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd
import os
from pathlib import Path
import pyarrow as pa
from pyarrow import csv
from timeit import default_timer as timer
from sklearn import preprocessing
from sklearn import metrics


def assert_exists(path: Path) -> None:
    """
    Assert that the path exists, else raise an exception
    :param path: directory as Path
    :return:
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist!")


def compare(directory: Path) -> Dict[str, Any]:
    source_utilization_parent = directory / "config"
    sample_utilization_parent = directory / "config_sampled"

    assert_exists(source_utilization_parent)
    assert_exists(sample_utilization_parent)

    source_utilization_dir = None
    for i, folder in enumerate(source_utilization_parent.iterdir()):
        # there should only be one folder
        if i >= 1:
            raise RuntimeError(f"More than one entry under the folder {source_utilization_parent}.")
        source_utilization_dir = folder

    if source_utilization_dir is None:
        raise RuntimeError(f"The folder {source_utilization_parent} is empty!")

    source_utilization_csv = None
    for i, file in enumerate(source_utilization_dir.glob("*.csv")):
        # there should also only be one source utilization csv file
        if i >= 1:
            raise RuntimeError(f"More than one utilization CSV under the folder {source_utilization_dir}.")
        source_utilization_csv = file

    if source_utilization_csv is None:
        raise RuntimeError(f"The folder {source_utilization_dir} is empty! The source CSV file is not found!")

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1_000_000)) # 1 million maybe? I'm worried about floating point errors

    # faster CSV read
    start = timer()
    source_utilization_df: pd.DataFrame = csv.read_csv(source_utilization_csv).to_pandas()
    source_utilization_ndarray = source_utilization_df.to_numpy()
    del source_utilization_df  # save memory

    normalized_source_utilization_ndarray = min_max_scaler.fit_transform(source_utilization_ndarray[:, [0]])  # first column, "Timestamp"
    normalized_source_utilization_ndarray = np.hstack((normalized_source_utilization_ndarray, min_max_scaler.fit_transform(source_utilization_ndarray[:, [1]])))  # hstack second column, "Core"
    normalized_source_utilization_ndarray = np.hstack((normalized_source_utilization_ndarray, min_max_scaler.fit_transform(source_utilization_ndarray[:, [2]])))  # hstack third column, "Memory"
    end = timer()
    print(f"Loaded source utilization CSV {source_utilization_csv} in {end-start} seconds.")

    json_output = dict()

    # run through each sampled util CSV and compare
    for sample_folder in sample_utilization_parent.iterdir():
        sample_utilization_csv = list(sample_folder.glob("*.csv"))[0]  # There should only be the one CSV file
        start = timer()
        sample_utilization_df: pd.DataFrame = csv.read_csv(sample_utilization_csv).to_pandas()
        sample_utilization_ndarray = sample_utilization_df.to_numpy()
        del sample_utilization_df
        end = timer()
        print(f"Loaded sample utilization CSV {sample_utilization_csv} in {end - start} seconds.")
        # normalize with min max
        # min max normalize for easier comparison (on both)
        normalized_sample_utilization_ndarray = min_max_scaler.fit_transform(sample_utilization_ndarray[:, [0]])  # first column, "Timestamp"
        normalized_sample_utilization_ndarray = np.hstack((normalized_sample_utilization_ndarray, min_max_scaler.fit_transform(sample_utilization_ndarray[:, [1]])))  # hstack second column, "Core"
        normalized_sample_utilization_ndarray = np.hstack((normalized_sample_utilization_ndarray, min_max_scaler.fit_transform(sample_utilization_ndarray[:, [2]])))  # hstack third column, "Memory"

        # compare the core and memory columns
        # since the utilization script does a running sum, i *think* that the proper way to quantify the difference between the two utilization sets
        # is to multiply the source sum by the sampled percentage and compare with the sum of the sample utilization set
        basename = sample_folder.name
        # find percentage from name
        i_start = basename.find(".")-1
        i_end = basename.find("_", i_start)
        if i_end < 0:
            i_end = basename.find(".", i_start)
        percentage = float(basename[i_start:i_end])

        # calculate source * percentage
        source_cores = sum(normalized_source_utilization_ndarray[:,1]) * percentage
        sampled_cores = sum(normalized_sample_utilization_ndarray[:,1])
        error_cores_sum = (sampled_cores - source_cores) / source_cores * 100
        source_memory = sum(normalized_source_utilization_ndarray[:, 2]) * percentage
        sampled_memory = sum(normalized_sample_utilization_ndarray[:, 2])
        error_memory_sum = (sampled_memory - source_memory) / source_memory * 100

        # AUC
        source_timestamps = normalized_source_utilization_ndarray[:, 0]
        source_cores = normalized_source_utilization_ndarray[:, 1]
        source_memory = normalized_source_utilization_ndarray[:, 2]
        sample_timestamps = normalized_sample_utilization_ndarray[:, 0]
        sample_cores = normalized_sample_utilization_ndarray[:, 1]
        sample_memory = normalized_sample_utilization_ndarray[:, 2]

        source_auc_cores = metrics.auc(source_timestamps, source_cores)  # trapezoidal
        source_auc_memory = metrics.auc(source_timestamps, source_memory)
        sample_auc_cores = metrics.auc(sample_timestamps, sample_cores)
        sample_auc_memory = metrics.auc(sample_timestamps, sample_memory)
        error_memory_auc = (sample_auc_memory - source_auc_memory) / source_auc_memory * 100
        error_cores_auc = (sample_auc_cores - source_auc_cores) / source_auc_cores * 100

        print(f"Core sum error: {error_cores_sum}% -- Memory sum error: {error_memory_sum}%")

        print(f"Core AUC error: {error_cores_auc}% -- Memory AUC error: {error_memory_auc}%")
        json_output.setdefault(str(percentage), [])
        json_output[str(percentage)].append({"core": {"sum": error_cores_sum, "auc": error_cores_auc}, "memory": {"sum": error_memory_sum, "auc": error_memory_auc}})
    return json_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default=None,
                        help="Directory to compare the files to. The base/source csv will be in DIR/config/FOLDER, "
                             "while the files to compare to will be in DIR/config_sampled/FOLDER")
    args = parser.parse_args()

    compare(Path(args.directory))


if __name__ == "__main__":
    main()

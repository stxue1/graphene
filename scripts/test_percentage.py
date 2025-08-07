import argparse
import sys
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import json


utilization_import = import_module("0-utilization")
sampling_import = import_module("1-sampling-azure-google")
run_utilization = getattr(utilization_import, "main")
run_sampling = getattr(sampling_import, "process_directory")
get_store_filename = getattr(utilization_import, "get_store_filename")
from compare import compare as run_compare


def sample_one_probability(probability: float, directory: Path, samples: int) -> None:
    # get_store_filename()

    # assume source CSV file must exist
    # sample first
    run_sampling(directory, probability, samples)


def generate_utilization(directory: Path, fresh_start: bool = False) -> None:

    # create utilization csv files
    if fresh_start:
        run_utilization(f"-d {directory} -a save".split())

    run_utilization(f"-d {directory} -a save --use-samples".split())

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Directory containing the trace files", default=None)
    parser.add_argument("-p", "--probability", help="Probability for when running Bernoulli Sampling over the trace files.",
                        default="0.08", type=str)
    parser.add_argument("-i", "--interval", help="Step to jump per probability while iterating over the given range (if provided)", type=Decimal)
    parser.add_argument("-s", "--samples", help="Number of samples to produce.", default=2, type=int)
    parser.add_argument("--fresh-start", help="If there is no CSV utilization file for the source CSV (unsampled) set this", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False, help="Don't run the scripts and just create a graph from the already ran CSV files")
    parser.add_argument("--sample-only", action="store_true", default=False, help="Only run the sampling and utilization scripts, don't graph/plot.")
    parser.add_argument("-a", "--action", default="plot", help="To save a plot as png or to graph it.", choices=["plot", "save"])
    parser.add_argument("--graph-type", "-t", default="std", choices=["std", "box"], help="Type of the graph to create (normal creates error bars based on standard deviation and box creates box plots).")
    options = parser.parse_args(args)

    if options.directory is None:
        print(f"Default directory to GoogleA...")
        options.directory = "GoogleA"
    to_write = Path(options.directory) / f"test_probability_{options.probability}.json"
    if not options.dry_run or options.sample_only:
        # process percentages
        probability = None
        start_probability = None
        end_probability = None
        probability_parse_type = None
        if options.probability.find("-") >= 0:
            ranges = list(map(Decimal, options.probability.split("-")))
            start_probability = ranges[0]
            end_probability = ranges[1]
            probability_parse_type = "range"
        elif options.probability.find(",") >= 0:
            probability_parse_type = "list"
        else:
            probability_parse_type = "one"
            probability = Decimal(options.probability)

        output_json = {}

        if probability_parse_type == "one":
            # run once on one probability
            sample_one_probability(float(probability), Path(options.directory), options.samples)
            generate_utilization(Path(options.directory), options.fresh_start)

            if options.sample_only:
                return

            output_json = run_compare(Path(options.directory))
        elif probability_parse_type == "range":
            if options.interval is None:
                raise RuntimeError("Provide an interval/step with the range with -i")
            current_probability = start_probability
            while current_probability < end_probability:
                sample_one_probability(float(current_probability), Path(options.directory), options.samples)
                current_probability += options.interval
            generate_utilization(Path(options.directory), options.fresh_start)

            if options.sample_only:
                return

            output_json.update(run_compare(Path(options.directory)))
        else:
            probability_list = options.probability.split(",")
            for p in probability_list:
                p = Decimal(p)
                sample_one_probability(float(p), Path(options.directory), options.samples)
            generate_utilization(Path(options.directory), options.fresh_start)

            if options.sample_only:
                return

            output_json.update(run_compare(Path(options.directory)))

        with open(to_write, "w") as f:
            json.dump(output_json, f)
    else:
        if options.sample_only:
            return
        with open(to_write, "r") as f:
            output_json = json.load(f)

    color = 'tab:red'
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    keys_in_order = sorted(list(map(float, output_json.keys())))

    prob_x = []
    df_per_prob = []
    for prob in keys_in_order:
        prob_x.append(prob)
        list_of_errors = output_json[str(prob)]
        df = pd.DataFrame(columns=["Error Core Sum", "Error Core AUC", "Error Memory Sum", "Error Memory AUC"])
        for i, error_dict in enumerate(list_of_errors):
            error_core_sum = error_dict["core"]["sum"]
            error_core_auc = error_dict["core"]["auc"]
            error_memory_sum = error_dict["memory"]["sum"]
            error_memory_auc = error_dict["memory"]["auc"]

            new_row = {"Error Core Sum": error_core_sum,
                       "Error Core AUC": error_core_auc,
                       "Error Memory Sum": error_memory_sum,
                       "Error Memory AUC": error_memory_auc}
            if len(df) == 0:
                # avoid warning
                df = pd.Series(new_row).to_frame().T
            else:
                df = pd.concat([df, pd.Series(new_row).to_frame().T], ignore_index=True)
        df_per_prob.append(df)

    if options.graph_type == "std":
        ax[0,0].set_title("Core % error (sum) / Prob (pt=mean)")
        ax[0,0].errorbar(prob_x, [np.mean(df["Error Core Sum"]) for df in df_per_prob], [np.std(df["Error Core Sum"]) for df in df_per_prob], linestyle='None', marker='.', ecolor=color)
        ax[1,0].set_title("Core % error (AUC) / Prob (pt=mean)")
        ax[1,0].errorbar(prob_x, [np.mean(df["Error Core AUC"]) for df in df_per_prob], [np.std(df["Error Core AUC"]) for df in df_per_prob], linestyle='None', marker='.', ecolor=color)
        ax[0,1].set_title("Mem % error (sum) / Prob (pt=mean)")
        ax[0,1].errorbar(prob_x, [np.mean(df["Error Memory Sum"]) for df in df_per_prob], [np.std(df["Error Memory Sum"]) for df in df_per_prob], linestyle='None', marker='.', ecolor=color)
        ax[1,1].set_title("Mem % error (AUC) / Prob (pt=mean)")
        ax[1,1].errorbar(prob_x, [np.mean(df["Error Memory AUC"]) for df in df_per_prob], [np.std(df["Error Memory AUC"]) for df in df_per_prob], linestyle='None', marker='.', ecolor=color)

        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlabel('Probability')
                ax[i, j].set_ylabel('% Error')
    else:
        ax[0, 0].set_title("Core % error (sum) / Prob")
        ax[0, 0].boxplot([df["Error Core Sum"] for df in df_per_prob])
        ax[1, 0].set_title("Core % error (AUC) / Prob")
        ax[1, 0].boxplot([df["Error Core AUC"] for df in df_per_prob])
        ax[0, 1].set_title("Mem % error (sum) / Prob")
        ax[0, 1].boxplot([df["Error Memory Sum"] for df in df_per_prob])
        ax[1, 1].set_title("Mem % error (AUC) / Prob")
        ax[1, 1].boxplot([df["Error Memory AUC"] for df in df_per_prob])

        ax[0, 0].set_xticks(range(1, len(prob_x) + 1), prob_x, rotation="vertical")
        ax[1, 0].set_xticks(range(1, len(prob_x) + 1), prob_x, rotation="vertical")
        ax[0, 1].set_xticks(range(1, len(prob_x) + 1), prob_x, rotation="vertical")
        ax[1, 1].set_xticks(range(1, len(prob_x) + 1), prob_x, rotation="vertical")

        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlabel('Probability')
                ax[i, j].set_ylabel('% Error')

    # plt.tick_params(axis='y', labelcolor=color)
    if options.action == "plot":
        plt.show()
    else:
        plt.savefig(str(to_write).replace('.json', '.png'))


if __name__ == "__main__":
    main()

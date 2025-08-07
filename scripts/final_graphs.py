import sys
import argparse
from importlib import import_module
import pandas as pd
import os
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams.update({
    "font.family": 'TeX Gyre Schola Math'
})
def classify_alibaba():
    # breakdown of size of scheduled jobs, pie chart, figure 0.1
    pie_chart_import = import_module("classify_alibaba_dag")
    pie_chart = getattr(pie_chart_import, "main")

    print("running classify alibaba")
    pie_chart("--trace-file Alibaba/trace/updated_vms.csv".split(" "))

def queueing_delay_over_instances_jointplot():
    # 1 million sample points jointplot graph of qdelay over instance num
    from coefficient import jointplot

    output_directory = "coefficient"
    input_csv = 'batch_instance_analysis/qdelay_pts_dag_1000000.csv' 
    df = pd.read_csv(input_csv)

    df = df[df['queueing_delay'] > 0]

    df = df.sort_values(by="timestamp")
    df.drop_duplicates(inplace=True)

    columns = ["timestamp", "remaining_cpu_cluster", "remaining_mem_cluster",
               "remaining_cpu_machine", "remaining_mem_machine"]
    remaining_usage_csv = 'machine_instance_analysis/remaining_machine_usage.csv'

    remaining_df = pd.read_csv(remaining_usage_csv, names=columns, header=None)

    merged_df = pd.merge_asof(df, remaining_df, on="timestamp", direction="backward")
    merged_df['queueing_delay'] = np.log(merged_df['queueing_delay'])
    merged_df['instance_num'] = np.log(merged_df['instance_num'])
    output = os.path.join(output_directory, "jointplot_queueing_delay_over_instance_num.png")
    jointplot(merged_df, "instance_num", "queueing_delay", f"Number of instances (log)", "Queueing delay in seconds (log)", output, title="Hexbin plot of queueing delay and instance number", y=0.95, top=0.9)
    def correlation_matrix(df, relevant, output, just_queueing_delay=False, mask=False, figsize=(4,2), fontsize=12):
        """
        Custom def from coefficient.py to avoid output filename munging
        """
        plt.figure(figsize=figsize)
        corr = df[relevant].corr()
        if mask:
            matrix_mask = ~np.triu(np.ones_like(corr, dtype=bool))
        else:
            matrix_mask = None
        if just_queueing_delay:
            corr = corr.loc[["queueing_delay"]]
        sns.heatmap(corr, mask=matrix_mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={"fontsize": fontsize})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output, dpi=300)
        plt.show()
    correlation_matrix(merged_df, ["queueing_delay", "instance_num"], "queueing_delay_over_instance_num_correlation.png", just_queueing_delay=True)

    # correlation metrix with everything else
    merged_df['plan_cpu'] = merged_df['plan_cpu'] * merged_df['instance_num']
    merged_df['plan_mem'] = merged_df['plan_mem'] * merged_df['instance_num']
    merged_df['plan_cpu'] = np.log(merged_df['plan_cpu'])
    merged_df['plan_mem'] = np.log(merged_df['plan_mem'])
    all_columns = [
        "queueing_delay",
        "cpu_avg",
        "cpu_max",
        "mem_avg",
        "mem_max",
        "at_req_cpu_subscribed",
        "at_req_mem_subscribed",
        # "at_req_lowest_machine_mem_util_percent",
        # "at_req_lowest_machine_cpu_util_percent",
        "at_req_highest_machine_mem_util_percent",
        "at_req_highest_machine_cpu_util_percent",
        "at_req_cluster_cpu_util_percent",
        "at_req_cluster_mem_util_percent",
        "plan_cpu",
        "plan_mem",
        "instance_num",
        "remaining_cpu_cluster",
        "remaining_mem_cluster",
        "remaining_cpu_machine",
        "remaining_mem_machine"
    ]
    # it may be useful also making a correlation matrix without log
    # there may be some dividebyzero errors, but I believe the difference is minimal
    correlation_matrix(merged_df, all_columns, "queueing_delay_correlations.png", figsize=(12,8), mask=True, fontsize=12)
    # example jointplot bewteen queueing delay and remaining total cluster memory
    merged_df['remaining_mem_cluster'] = np.log(merged_df['remaining_mem_cluster'])
    output = "jointplot_queueing_delay_over_remaining_mem.png"
    jointplot(merged_df, "remaining_mem_cluster", "queueing_delay", f"Remaining cluster memory (log)", "Queueing delay in seconds (log)", output, title="Hexbin Plot of Queueing Delay and Remaining Total Memory", top=0.9, y=0.95)
    
def queueing_delay_cdf():
    # quueeing delay cdf figure 0.2
    # over a sample of 1 million points
    # batch_instance_analysis/queueing_delays_completion_times_dag_cdf.png
    qdelay_import = import_module("batch_instance_analyze")
    qdelay = getattr(qdelay_import, "graph_queueing_delay")
    # filename is a little misleading
    # critical_path_qdelay.py::plot_cdf_log_x( is responsible for plot formatting
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", "-t", help="Where the machine_usage trace is", default="../alibaba-trace/batch_instance.csv")
    parser.add_argument("--operation", "-o", help="The operation to run on the trace")
    parser.add_argument("--output", help="output directory", default="batch_instance_analysis")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--norm", default=False, action="store_true", help="Whether to normalize data when graphing")
    parser.add_argument("--infile", default=None)
    parser.add_argument("--instance-to-task-estimation", "-e", help="What algorithm to use when converting instance level data to task level data. Ex: max, avg",
                        default="avg")
    parser.add_argument("--lookup-job")
    parser.add_argument("--lookup-task")
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--num-points", type=int, default=1_000_000)
    parser.add_argument("--dag", action="store_true", default=False)
    options = parser.parse_args()

    qdelay(options)

def cluster_utilization():
    from cluster_analyze import graph_cluster_utilization
    parser = argparse.ArgumentParser()
    options = parser.parse_args()
    graph_cluster_utilization(options)
    

def queueing_delay_over_completion():
    from graph_queueing_delay import graph_queueing_delay_over_completion
    parser = argparse.ArgumentParser(description="Run a specified operation.")
    parser.add_argument("--operation", "-o", type=str, help="Name of the operation (function) to execute.")
    parser.add_argument("--overwrite", action="store_true", default=-False)
    parser.add_argument("--num-points", "-n", type=int)
    parser.add_argument("--dag", type=bool, default=True)
    options = parser.parse_args()
    graph_queueing_delay_over_completion(options)

def critical_queueing_delay_over_completion():
    from critical_path_qdelay import plot_graphs
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", "-n", type=int)
    options = parser.parse_args()
    plot_graphs(options)
def oversubscription_hist(data_type='mem_avg'):
    # code taken from batch_instance_analyze.py::graph_oversubscription
    num_points = 1_000_000
    cache_file = os.path.join("oversubscription_analysis", f"{data_type}_{num_points}.csv")
    sampled_data_type = pd.read_csv(cache_file)['points']
    
    from batch_instance_analyze import plot_hist_log
    plot_hist_log(data=sampled_data_type, n_bins=100, title=f"Oversubscription {data_type} (reciprocal)",
                  output=os.path.join("oversubscription_analysis", f"oversubscription_{data_type}.png"),
                  xlabel="Log Oversubscription Factor", ylabel="Count")
def machineids():
    path = "timeline/batch_job_task_name_j_2057344.csv"
    df = pd.read_csv(path)
    machine_id_counts = df["machine_id"].value_counts()

    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    n_bins = len(np.unique(machine_id_counts))
    color = "slateblue"
    ax1.hist(machine_id_counts, bins=n_bins, color=color)
    ax1.set_title("Repeat Machine Allocations")
    ax1.set_xlabel("Number of repeat allocations")
    ax1.set_ylabel("Count")
    output = "j_2057344_machineids.png"
    plt.savefig(output)
    plt.clf()

def all_machineids():
    path = "timeline/metrics.csv"
    df = pd.read_csv(path)
    fig, ax1 = plt.subplots(tight_layout=True)
    machine_id_counts = df["count"]
    n_bins = 1000
    color = "slateblue"
    ax1.hist(machine_id_counts, bins=n_bins, color=color)
    ax1.set_title("Repeat Machine Allocations")
    ax1.set_xlabel("Number of repeat allocations")
    ax1.set_ylabel("Count")
    output = "all_machineids.png"
    plt.savefig(output)
    plt.clf()

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()

    options = parser.parse_args(args)

    # classify_alibaba()
    # queueing_delay_cdf()
    queueing_delay_over_instances_jointplot()
    # queueing_delay_over_completion()
    # cluster_utilization()
    # critical_queueing_delay_over_completion()
    # oversubscription_hist('mem_avg')
    # oversubscription_hist('cpu_avg')
    # machineids()
    # all_machineids()

if __name__ == "__main__":
    main()

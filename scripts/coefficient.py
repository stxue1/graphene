#!/usr/bin/env python3
"""
This script reads a trace CSV file with header:
    Start Time,End Time,Core,Memory,Min_ratio,Coeff
and produces several plots:
  1. A scatter plot grouping jobs by (rounded) CPU and Memory.
     - X axis: Memory (rounded)
     - Y axis: CPU (rounded)
     - The circle’s area (and color) represents the frequency (number of jobs).
  2. A scatter plot showing the correlation between Memory and job lifetime.
  3. A histogram of job lifetimes.
  4. A hexbin jointplot between Memory and Lifetime.
  5. A correlation matrix heatmap among several variables.
If your CSV file is huge (~10GB), consider using Dask or reading in chunks.
"""
import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from graph_queueing_delay import plot_scatter_with_job_length

# Uncomment the following lines if you prefer to use Dask for large CSV files.
# import dask.dataframe as dd
# def load_csv_dask(filename):
#     # Read CSV with dask (this returns a lazy dataframe)
#     ddf = dd.read_csv(filename)
#     # Compute the dataframe (you may want to use further processing on ddf before computing)
#     return ddf.compute()
#
# def load_csv_pandas(filename):
#     return pd.read_csv(filename)
output_directory = "coefficient"


def graph_correlation(xaxis, yaxis, xaxis_label, yaxis_label, jxaxis, jyaxis, jxaxis_label, jyaxis_label, relevant):
    # ==========================
    # PARAMETERS & FILE PATHS
    # ==========================
    os.makedirs(output_directory, exist_ok=True)
    input_csv = 'batch_instance_analysis/qdelay_pts_dag_1000000.csv'  # Change this to your CSV file path
    # ==========================
    # LOAD THE DATA
    # ==========================
    print("Loading CSV file...")
    try:
        # If you have enough RAM, you can use Pandas directly:
        df = pd.read_csv(input_csv)
        # If you need to use Dask, comment out the above line and uncomment:
        # df = load_csv_dask(input_csv)
    except Exception as e:
        print("Error loading CSV:", e)
        return
    print(f"CSV loaded successfully. Number of jobs: {len(df)}")
    # Remove negative qdelay
    df = df[df['queueing_delay'] > 0]
    print(f"CSV filtered successfully. Number of jobs: {len(df)}")
    # Clean up the column names (remove extra spaces)
    df.columns = [col.strip() for col in df.columns]
    # ---------------------------
    # Compute job lifetime
    # ---------------------------
    # df['Lifetime'] = df['End Time'] - df['Start Time']
    # ===========================================
    # 1. Grouping Jobs by Resource (CPU vs Memory)
    # ===========================================
    # Here we round the "Core" and "Memory" values so that jobs with similar
    # resource requirements fall into the same bin.
    jxaxis_round = f'{jxaxis}_round'
    jyaxis_round = f'{jyaxis}_round'

    df[jxaxis_round] = df[jxaxis].astype(float)
    # if jyaxis is cpu, consider rounding to int
    df[jyaxis_round] = df[jyaxis].astype(float)

    # log the relevant values
    # df['queueing_delay'] = np.log(df['queueing_delay'])
    # df['mem_avg'] = np.log(df['mem_avg'])
    # df['mem_max'] = np.log(df['mem_max'])
    # df['cpu_avg'] = np.log(df['cpu_avg'])
    # df['cpu_max'] = np.log(df['cpu_max'])

    # Group by the rounded values and count the frequency
    grouped = df.groupby([jxaxis_round, jyaxis_round]).size().reset_index(name='count')
    print("Jobs grouped by resources. Number of unique (cpu_avg, mem_avg) bins:", len(grouped))
    # ---------------------------
    # Plot 1: Scatter plot of job shapes
    # ---------------------------
    xlog = True
    ylog = True

    plt.figure(figsize=(12, 8))
    # Scale the marker sizes so that the largest group gets a marker of size 100 (adjust factor as needed)
    sizes_scaled = 100 * (grouped['count'] / grouped['count'].max())
    scatter = plt.scatter(grouped[jxaxis_round], grouped[jyaxis_round],
                          s=sizes_scaled, alpha=0.6,
                          c=grouped['count'], cmap='viridis', edgecolors='w', linewidth=0.5)
    plt.xlabel(jxaxis_label)
    plt.ylabel(jyaxis_label)
    # if xlog:
    #     plt.xscale('log')
    # if ylog:
    #     plt.yscale('log')
    plt.title('Job Frequency by cpu_avg and mem_avg')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Job Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'D-job_resource_distribution.png'), dpi=300)
    plt.show()
    # ===========================================
    # 2. Correlation between Memory and Job qdleay
    # ===========================================
    plt.figure(figsize=(12, 8))
    plt.scatter(df[xaxis], df[yaxis], alpha=0.3, color='teal', edgecolors='none')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    # if xlog:
    #     plt.xscale('log')
    # if ylog:
    #     plt.yscale('log')
    plt.title('Correlation between Memory and Job qdelay')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'D-memory_vs_qdelay.png'), dpi=300)
    plt.show()
    # ===========================================
    # Additional Plots for Job Characterization
    # ===========================================
    # a) Histogram of Job qdelay
    plt.figure(figsize=(12, 8))
    plt.hist(df[yaxis], bins=100, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel(yaxis_label)
    plt.ylabel('Frequency')
    # if xlog:
    #     plt.xscale('log')
    # if ylog:
    #     plt.yscale('log')
    plt.title('Distribution of Job qdelays')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'D-job_qdelay_histogram.png'), dpi=300)
    plt.show()
    # b) Jointplot (hexbin) between Memory and Lifetime using seaborn
    # This gives you a sense of density in the 2D space.
    joint_plot = sns.jointplot(x=xaxis, y=yaxis, data=df, kind='hex', bins=1000, height=8)
    joint_plot.fig.suptitle(f'{xaxis_label} vs. {yaxis_label} (Hexbin Jointplot)', y=1.02)
    # if xlog:
    #     plt.xscale('log')
    # if ylog:
    #     plt.yscale('log')
    plt.savefig(os.path.join(output_directory, 'D-memory_qdelay_jointplot.png'), dpi=300)
    plt.show()

    correlation_matrix(df, relevant, output_directory)


def just_correlation_matrix(relevant):
    input_csv = 'batch_instance_analysis/qdelay_pts_dag_1000000.csv'  # Change this to your CSV file path
    df = pd.read_csv(input_csv)
    # df.drop_duplicates(subset=["job_name", "task_name"])
    df = df[df['queueing_delay'] > 0]
    df = df[df['mem_avg'] > 0]
    df = df[df['mem_max'] > 0]
    df = df[df['cpu_avg'] > 0]
    df = df[df['cpu_max'] > 0]

    # log the relevant values
    df['queueing_delay'] = np.log(df['queueing_delay'])
    # df['completion_time'] = np.log(df['completion_time'])
    df['oversubscription_cpu_avg'] = np.log(df['plan_cpu'] / df['cpu_avg'])
    df['oversubscription_mem_avg'] = np.log(df['plan_mem'] / df['mem_avg'])

    df['plan_cpu'] = df['plan_cpu'] * df['instance_num']
    df['plan_mem'] = df['plan_mem'] * df['instance_num']

    # df['instance_num'] = np.log(df['instance_num'])
    # df['plan_cpu'] = np.log(df['plan_cpu'])
    # df['plan_mem'] = np.log(df['plan_mem'])
    # df['mem_avg'] = np.log(df['mem_avg'])
    # df['mem_max'] = np.log(df['mem_max'])
    # df['cpu_avg'] = np.log(df['cpu_avg'])
    # df['cpu_max'] = np.log(df['cpu_max'])
    correlation_matrix(df, relevant, output_directory)

def jointplot(df, xaxis, yaxis, xlabel, ylabel, output, bins=1000, title=None, y=None, top=None):
    joint_plot = sns.jointplot(x=xaxis, y=yaxis, data=df, kind='hex', color="royalblue", bins=bins, height=6)
    # joint_plot.fig.suptitle(f'{xlabel} vs. {ylabel} (Hexbin Jointplot)', y=1.02)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if title:
        plt.suptitle(title, y=y or 1)
        # joint_plot.fig.suptitle(title, y=y or 1)
        plt.subplots_adjust(top=top or 1.0)
    plt.savefig(output, dpi=300)
    plt.show()
def correlation_matrix_test():
    def remaining_plots(merged_df, output_directory):
        merged_df['queueing_delay'] = np.log(merged_df['queueing_delay'])
        merged_df['remaining_mem_cluster'] = np.log(merged_df['remaining_mem_cluster'])
        merged_df['remaining_cpu_cluster'] = np.log(merged_df['remaining_cpu_cluster'])

        output = os.path.join(output_directory, "queueing_delay_over_remaining_mem.png")
        plot_scatter_with_job_length(merged_df["remaining_mem_cluster"], merged_df["queueing_delay"], merged_df["job_length"],
                                     xlabel="remaining_mem_cluster (log)", ylabel="queueing delay (log)", output=output, log=False)
        output = os.path.join(output_directory, "jointplot_queueing_delay_over_remaining_mem.png")
        jointplot(merged_df, "remaining_mem_cluster", "queueing_delay", "remaining_mem_cluster (log)", "queueing_delay (log)", output)
        output = os.path.join(output_directory, "queueing_delay_over_remaining_cpu.png")
        plot_scatter_with_job_length(merged_df["remaining_cpu_cluster"], merged_df["queueing_delay"], merged_df["job_length"],
                                     xlabel="remaining_cpu_cluster (log)", ylabel="queueing delay (log)", output=output, log=False)
        output = os.path.join(output_directory, "jointplot_queueing_delay_over_remaining_cpu.png")
        jointplot(merged_df, "remaining_cpu_cluster", "queueing_delay", "remaining_cpu_cluster (log)", "queueing_delay (log)", output)
        correlation_matrix(merged_df, ["queueing_delay", "remaining_mem_cluster", "remaining_cpu_cluster"], output_directory)


    def ratio_plots(merged_df, output_directory, typ="mem"):
        field = f'ratio_req_over_remaining_{typ}'
        merged_df[field] = (merged_df[f'plan_{typ}'] * merged_df['instance_num']) / merged_df[f'remaining_{typ}_cluster']

        output = os.path.join(output_directory, f"scatter_queueing_delay_over_{field}.png")
        plot_scatter_with_job_length(merged_df[field], merged_df['queueing_delay'], merged_df['completion_time'], xlabel=f"{field} (log)", ylabel="queueing_delay (log)",
                                     output=output, log=True)

        merged_df['queueing_delay'] = np.log(merged_df['queueing_delay'])
        merged_df[field] = np.log(merged_df[field])
        # correlation_matrix(merged_df, ["queueing_delay", field], output_directory)
        output = os.path.join(output_directory, f"jointplot_queueing_delay_over_{field}.png")
        jointplot(merged_df, field, "queueing_delay", f"{field} (log)", "queueing_delay (log)", output)

    input_csv = 'batch_instance_analysis/qdelay_pts_dag_1000000.csv'  # Change this to your CSV file path
    # input_csv = 'critical_path_analysis/queueing_delays_critical_1000000.csv'
    columns = ["queueing_delay","cpu_avg","cpu_max","mem_avg","mem_max","at_req_cpu_subscribed","at_req_mem_subscribed","at_req_lowest_machine_cpu_util_percent","at_req_lowest_machine_mem_util_percent","at_req_highest_machine_cpu_util_percent","at_req_highest_machine_mem_util_percent","at_req_cluster_cpu_util_percent","at_req_cluster_mem_util_percent","plan_cpu","plan_mem","instance_num"]
    df = pd.read_csv(input_csv)
    # df.rename(columns={"start_time": "timestamp"}, inplace=True)

    df = df[df['queueing_delay'] > 0]
    # df.drop_duplicates(subset=["job_name", "task_name"], inplace=True)
    df['queueing_delay'] = df['queueing_delay'] + 1
    df['completion_time'] = df['completion_time'] + 1
    df = df[df['mem_avg'] > 0]
    df = df[df['mem_max'] > 0]
    df = df[df['cpu_avg'] > 0]
    df = df[df['cpu_max'] > 0]

    df.drop_duplicates(inplace=True)

    remaining_columns = ["timestamp", "remaining_cpu_cluster", "remaining_mem_cluster",
               "remaining_cpu_machine", "remaining_mem_machine"]
    remaining_usage_csv = 'machine_instance_analysis/remaining_machine_usage.csv'

    remaining_df = pd.read_csv(remaining_usage_csv, names=remaining_columns, header=None)

    df['scheduleable_time'] = df['timestamp'] - df['queueing_delay']
    df = df.sort_values(by="scheduleable_time")
    merged_df = pd.merge_asof(df, remaining_df, left_on="scheduleable_time", right_on="timestamp", direction="backward")

    # todo: unlog
    # for c in columns + remaining_columns[1:]:
    #     merged_df[c] = np.log(merged_df[c])
    # correlation_matrix(merged_df, ["queueing_delay", remaining_columns[1], remaining_columns[2], remaining_columns[3], remaining_columns[4]], output_directory)
    # correlation_matrix(merged_df, columns + remaining_columns[1:], output_directory)
    remaining_plots(merged_df.copy(), output_directory)

    ratio_plots(merged_df.copy(), output_directory)

    typ = "mem"
    field = f'plan_mem_total'

    merged_df[field] = merged_df['instance_num'] * merged_df['plan_mem']
    # merged_df[field] = (merged_df[f'plan_{typ}'] * merged_df["instance_num"])/ (merged_df[f'remaining_{typ}_cluster'])

    merged_df['queueing_delay'] = np.log(merged_df['queueing_delay'])
    merged_df[field] = np.log(merged_df[field])
    merged_df['instance_num'] = np.log(merged_df['instance_num'])

    output = os.path.join(output_directory, f"scatter_queueing_delay_over_{field}.png")
    plot_scatter_with_job_length(merged_df[field], merged_df['queueing_delay'], np.log(merged_df['completion_time']), xlabel=f"{field} (log)", ylabel="queueing_delay (log)",
                                 output=output, log=False)
    # correlation_matrix(merged_df, ["queueing_delay", "instance_num"], output_directory)
    output = os.path.join(output_directory, f"jointplot_queueing_delay_over_{field}.png")
    jointplot(merged_df, field, "queueing_delay", f"{field} (log)", "queueing_delay (log)", output)

    # plot_scatter_with_job_length(merged_df["instance_num"], merged_df["queueing_delay"], merged_df["job_length"],
    #                              xlabel="remaining_cpu_cluster (log)", ylabel="queueing delay (log)", output=output, log=False)
    # output = os.path.join(output_directory, "jointplot_queueing_delay_over_instance_num.png")
    # jointplot(merged_df, "instance_num", "queueing_delay", f"instance_num (log)", "queueing_delay (log)", output)


def correlation_matrix_critical():
    input_csv = 'critical_path_analysis/queueing_delays_critical_1000000.csv'
    df = pd.read_csv(input_csv)
    df.rename(columns={"start_time": "timestamp"}, inplace=True)

    df = df[df['queueing_delay'] > 0]

    df = df.sort_values(by="timestamp")
    df.drop_duplicates(inplace=True)

    columns = ["timestamp", "remaining_cpu_cluster", "remaining_mem_cluster",
               "remaining_cpu_machine", "remaining_mem_machine"]
    remaining_usage_csv = 'machine_instance_analysis/remaining_machine_usage.csv'

    remaining_df = pd.read_csv(remaining_usage_csv, names=columns, header=None)

    merged_df = pd.merge_asof(df, remaining_df, on="timestamp", direction="backward")

    typ = "mem"
    field = f'plan_mem'
    merged_df[field] = (merged_df[f'remaining_{typ}_cluster']) / (merged_df[f'plan_{typ}'])
    # merged_df[field] = (merged_df[f'plan_{typ}'])

    merged_df['queueing_delay'] = np.log(merged_df['queueing_delay'])
    merged_df[field] = np.log(merged_df[field])

    output = os.path.join(output_directory, f"scatter_critical_queueing_delay_over_{field}.png")
    plot_scatter_with_job_length(merged_df[field], merged_df['queueing_delay'], np.log(merged_df['job_length']), xlabel=f"{field} (log)", ylabel="queueing_delay (log)",
                                 output=output, log=False)
    correlation_matrix(merged_df, ["queueing_delay", field], output_directory)
    output = os.path.join(output_directory, f"jointplot_critical_queueing_delay_over_{field}.png")
    jointplot(merged_df, field, "queueing_delay", f"{field} (log)", "queueing_delay (log)", output)

def correlation_matrix_oversubscription():
    input_csv = 'batch_instance_analysis/qdelay_pts_dag_1000000.csv'
    df = pd.read_csv(input_csv)
    df.drop_duplicates(subset=["job_name", "task_name"], inplace=True)

    df.rename(columns={"start_time": "timestamp"}, inplace=True)

    df = df[df['queueing_delay'] > 0]

    df = df.sort_values(by="timestamp")
    df.drop_duplicates(inplace=True)

    oversubscription_csv = 'oversubscription_analysis/running_usage.csv'

    oversubscription_df = pd.read_csv(oversubscription_csv)
    oversubscription_df.sort_values(by="timestamp", inplace=True)

    # todo: this isn't correct, I need to match by job and task name instead of timestamp
    # merged_df = pd.merge_asof(df, oversubscription_df, on="timestamp", direction="backward")
    merged_df = pd.merge(df, oversubscription_df, on=["job_name", "task_name"])

    typ = "mem"
    field = f'oversubscription_mem_avg'
    # merged_df = merged_df[merged_df[field] > 0]
    # merged_df[field] = (merged_df[f'remaining_{typ}_cluster']) / (merged_df[f'plan_{typ}'])

    merged_df['queueing_delay'] = np.log(merged_df['queueing_delay'])
    merged_df[field] = np.log(merged_df[field] + merged_df[field].min() + 1)

    output = os.path.join(output_directory, f"scatter_queueing_delay_over_{field}.png")
    plot_scatter_with_job_length(merged_df[field], merged_df['queueing_delay'], np.log(merged_df['job_length']), xlabel=f"{field} (log)", ylabel="queueing_delay (log)",
                                 output=output, log=False)
    correlation_matrix(merged_df, ["queueing_delay", field], output_directory)
    output = os.path.join(output_directory, f"jointplot_queueing_delay_over_{field}.png")
    jointplot(merged_df, field, "queueing_delay", f"{field} (log)", "queueing_delay (log)", output)
def correlation_matrix(df, relevant, output_directory):
    # c) Correlation Matrix Heatmap (between a few numerical columns)
    plt.figure(figsize=(10, 8))
    # Select a few columns that may be of interest
    corr = df[relevant].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"fontsize": 12})
    # if xlog:
    #     plt.xscale('log')
    # if ylog:
    #     plt.yscale('log')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'D-correlation_matrix.png'), dpi=300)
    plt.show()

def correlation_matrix_top():
    input_csv = 'batch_instance_analysis/qdelay_pts_dag_1000000.csv'  # Change this to your CSV file path
    # input_csv = 'critical_path_analysis/queueing_delays_critical_1000000.csv'
    df = pd.read_csv(input_csv)
    # df.rename(columns={"start_time": "timestamp"}, inplace=True)

    df = df[df['queueing_delay'] > 0]
    # df.drop_duplicates(subset=["job_name", "task_name"], inplace=True)
    # df['queueing_delay'] = df['queueing_delay'] + 1
    # df['completion_time'] = df['completion_time'] + 1
    # df = df[df['mem_avg'] > 0]
    # df = df[df['mem_max'] > 0]
    # df = df[df['cpu_avg'] > 0]
    # df = df[df['cpu_max'] > 0]

    df = df.sort_values(by="timestamp")
    df.drop_duplicates(inplace=True)

    columns = ["timestamp", "remaining_cpu_cluster", "remaining_mem_cluster",
               "remaining_cpu_machine", "remaining_mem_machine"]
    remaining_usage_csv = 'machine_instance_analysis/remaining_machine_usage.csv'

    remaining_df = pd.read_csv(remaining_usage_csv, names=columns, header=None)

    merged_df = pd.merge_asof(df, remaining_df, on="timestamp", direction="backward")

    typ = "mem"
    field = f'plan_mem_total'

    merged_df[field] = merged_df['instance_num'] * merged_df['plan_mem']

    sorted_df = merged_df.sort_values(by='queueing_delay', ascending=False)
    top_10_percent_count = int(len(sorted_df) * 0.1)
    merged_df = sorted_df.head(top_10_percent_count).copy()
    merged_df = merged_df.iloc[:, 1:].copy()
    merged_df.to_csv(os.path.join(output_directory, "sorted.csv"), index=False)

    # merged_df[field] = (merged_df[f'plan_{typ}'] * merged_df["instance_num"])/ (merged_df[f'remaining_{typ}_cluster'])

    merged_df['queueing_delay'] = np.log(merged_df['queueing_delay'])
    merged_df[field] = np.log(merged_df[field])
    merged_df['instance_num'] = np.log(merged_df['instance_num'])

    output = os.path.join(output_directory, f"scatter_queueing_delay_over_{field}_top_10.png")
    plot_scatter_with_job_length(merged_df[field], merged_df['queueing_delay'], np.log(merged_df['completion_time']), xlabel=f"{field} (log)", ylabel="queueing_delay (log)",
                                 output=output, log=False)
    # correlation_matrix(merged_df, ["queueing_delay", "instance_num"], output_directory)
    output = os.path.join(output_directory, f"jointplot_queueing_delay_over_{field}_top_10.png")
    jointplot(merged_df, field, "queueing_delay", f"{field} (log)", "queueing_delay (log)", output)

    # plot_scatter_with_job_length(merged_df["instance_num"], merged_df["queueing_delay"], merged_df["job_length"],
    #                              xlabel="remaining_cpu_cluster (log)", ylabel="queueing delay (log)", output=output, log=False)
    # output = os.path.join(output_directory, "jointplot_queueing_delay_over_instance_num.png")
    # jointplot(merged_df, "instance_num", "queueing_delay", f"instance_num (log)", "queueing_delay (log)", output)



def main():
    # graph_correlation("mem_avg", "queueing_delay", xaxis_label="mem_avg", yaxis_label="Job qdelay", jxaxis="mem_avg", jyaxis="cpu_avg",
    #                   jxaxis_label="mem_avg", jyaxis_label="cpu_avg", relevant=['cpu_avg', 'cpu_max', 'mem_avg', 'mem_max', 'queueing_delay',
    #                                                                             'at_req_cpu_subscribed', 'at_req_mem_subscribed', 'at_req_lowest_machine_cpu_util_percent',
    #                                                                             'at_req_lowest_machine_mem_util_percent'])

    # just_correlation_matrix(relevant=[
    #     # 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max',
    #     'queueing_delay',
    #     'at_req_cpu_subscribed', 'at_req_mem_subscribed',
    #     #'at_req_lowest_machine_cpu_util_percent',
    #     # 'at_req_lowest_machine_mem_util_percent', 'at_req_highest_machine_cpu_util_percent', 'at_req_highest_machine_mem_util_percent',
    #     'at_req_cluster_cpu_util_percent', 'at_req_cluster_mem_util_percent', 'plan_cpu', 'plan_mem', 'instance_num',
    #     'completion_time',
    #     'oversubscription_cpu_avg',
    #     'oversubscription_mem_avg'
    # ])

    correlation_matrix_test()
    # correlation_matrix_critical()
    # correlation_matrix_top()


if __name__ == '__main__':
    main()

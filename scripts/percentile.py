import os
import sys
from importlib import import_module
import argparse
from pathlib import PurePath, Path
from typing import List, Union, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

plot_import = import_module("3-plot")
calculate_cdf = getattr(plot_import, "calculate_cdf")


def all_i(ls: List[Union[List[float], ArrayLike]], i: int) -> List[float]:
    return [l[i] for l in ls]


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-directory", "-d", dest="directory", required=True)
    parser.add_argument("--sample-count", "-s", dest="samples", type=int, required=True)
    parser.add_argument("--policy")
    parser.add_argument("--average", "--avg", "-a", dest="avg", action="store_true", default=False)
    parser.add_argument("--dont-show", dest="show", action="store_false", default=True, help="Whether to not show the plot and only write to disk.")
    parser.add_argument("--memory", dest="memory", help="Memory percentile to use.", required=True, type=float)
    parser.add_argument("--output", help="Optional filename to use. Must end in pdf.", default=None)

    options = parser.parse_args(args)

    all_directories = list()

    base_directory_path = PurePath(options.directory)

    basename = base_directory_path.name
    parent = base_directory_path.parent

    idx = basename.rfind("_")
    start_filename = basename[:idx]
    # sample_num = int(basename[idx+1:])

    for i in range(1, 1 + options.samples):
        all_directories.append(Path(parent) / Path(start_filename + "_" + str(i)))

    for directory in all_directories:
        if not os.path.exists(directory):
            raise RuntimeError("Directory %s does not exist!" % str(directory))

    titles = list()
    csv_files = list()
    for dir in all_directories:
        for csv in os.listdir(dir):
            csv_path = dir / Path(csv)
            title = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))  # todo: replace with better impl
            title = title.split("_")[-1]
            base = os.path.basename(csv)
            start = base.find("updated_vms") + len("updated_vms")
            memory_start = base.find("multiple_machines_memory")
            specifiers = base[start + 1:memory_start - 1].split("_")
            memory_specifier = base[memory_start + len("multiple_machines_memory") + 1:].split("_")[0]
            if options.memory is not None:
                if float(memory_specifier) != options.memory:
                    continue
            if "adaptive" == specifiers[0]:
                title = specifiers[0] + "_" + specifiers[1]
                if len(specifiers) >= 3:
                    title += "_" + specifiers[2]
                title += "_" + title
            else:
                title = "bernoulli" + "_" + specifiers[0]
                if len(specifiers) >= 2:
                    title += "_" + specifiers[1]
                title += "_" + title

            titles.append(title)
            csv_files.append(csv_path)
            policy = title.split("_")[1]

    xs_insufficient = list()
    ys_insufficient = list()
    xs_fragmented = list()
    ys_fragmented = list()
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)

        mem_cluster_bound_delay = df['mem cluster'] / (df['Finish Time'] - df['Arrival Time'])
        mem_machine_bound_delay = df['mem machine'] / (df['Finish Time'] - df['Arrival Time'])

        x_mem_cluster, y_mem_cluster = calculate_cdf(mem_cluster_bound_delay.dropna())
        x_mem_machine, y_mem_machine = calculate_cdf(mem_machine_bound_delay.dropna())

        insufficient_percentile_x = list()  # p of insufficient mem
        insufficient_percentile_y = list()  # q delay
        fragmented_percentile_x = list()  # p of fragmented mem
        fragmented_percentile_y = list()  # q delay

        for i in range(100):
            p = np.percentile(y_mem_cluster, i, method="higher")  # favor the higher percentile value
            idx = np.where(y_mem_cluster == p)[0][0]
            q_delay = x_mem_cluster[idx]
            insufficient_percentile_x.append(i)
            insufficient_percentile_y.append(q_delay)

            p = np.percentile(y_mem_machine, i, method="higher")  # also favor higher percentile value
            idx = np.where(y_mem_machine == p)[0][0]
            q_delay = x_mem_machine[idx]
            fragmented_percentile_x.append(i)
            fragmented_percentile_y.append(q_delay)

        # ax.plot(insufficient_percentile_x, insufficient_percentile_y, label=f'{titles[i]} Insufficient-Mem-{memory_specifier}', color=colors[0], linestyle=':', linewidth=8)
        # ax.plot(fragmented_percentile_x, fragmented_percentile_y, label=f'{titles[i]} Fragmented-Mem-{memory_specifier}', color=colors[1], linestyle='-', linewidth=8)

        xs_insufficient.append(insufficient_percentile_x)
        ys_insufficient.append(insufficient_percentile_y)
        xs_fragmented.append(fragmented_percentile_x)
        ys_fragmented.append(fragmented_percentile_y)

        # plt.show()
        # plt.close()

    xs_ins_interp = list()  # insufficient memory percentile
    ys_ins_interp = list()  # queueing delay for insufficient memory
    xs_frag_interp = list()
    ys_frag_interp = list()
    fidelity = 100
    for i in range(len(xs_insufficient)):
        xp_ins = xs_insufficient[i]
        yp_ins = ys_insufficient[i]
        xp_frag = xs_fragmented[i]
        yp_frag = ys_fragmented[i]

        min_x, max_x = min(xp_ins), max(xp_ins)
        new_xp_ins = np.linspace(min_x, max_x, fidelity)
        new_yp_ins = np.interp(new_xp_ins, xp_ins, yp_ins)

        min_x, max_x = min(xp_frag), max(xp_frag)
        new_xp_frag = np.linspace(min_x, max_x, fidelity)
        new_yp_frag = np.interp(new_xp_frag, xp_frag, yp_frag)

        xs_ins_interp.append(new_xp_ins)
        ys_ins_interp.append(new_yp_ins)
        xs_frag_interp.append(new_xp_frag)
        ys_frag_interp.append(new_yp_frag)

    avg_xp_ins = [np.mean(all_i(xs_ins_interp, i)) for i in range(fidelity)]
    avg_yp_ins = [np.mean(all_i(ys_ins_interp, i)) for i in range(fidelity)]

    avg_xp_frag = [np.mean(all_i(xs_frag_interp, i)) for i in range(fidelity)]
    avg_yp_frag = [np.mean(all_i(ys_frag_interp, i)) for i in range(fidelity)]

    fig, ax = plt.subplots(figsize=(14, 14))
    colors = ['#F1C40F', '#A569BD', '#3498DB', '#1ABC9C']
    # memory_start = str(csv_file).find("multiple_machines_memory")
    # memory_specifier = str(csv_file)[memory_start + len("multiple_machines_memory") + 1:].split("_")[0]
    memory_specifier = str(options.memory)
    title = f"q-delay / p — avg mem-{memory_specifier} n={options.samples}"
    x_label = f"Percentile"
    y_label = f"Queueing delay"
    ax.plot(avg_xp_ins, avg_yp_ins, label="Insufficient memory", color=colors[0], linestyle=':', linewidth=8)
    ax.plot(avg_xp_frag, avg_yp_frag, label="Fragmented memory", color=colors[1], linestyle='-', linewidth=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(top=1)
    plt.title(title)
    ax.set_xticks(np.arange(0, 110, step=10))

    plt.legend(loc="upper left")

    if options.show:
        plt.show()
        plt.close()

    output_path = f"percentile_graph_{memory_specifier}.pdf" if options.output is None else options.output
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved file to {output_path}")


if __name__ == "__main__":
    main()

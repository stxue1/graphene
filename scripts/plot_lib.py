import time
import logging
from matplotlib import pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def plot_hist_log_count(data, n_bins, title, output, xlabel, ylabel):

    start = time.time()
    plt.figure(figsize=(10, 10))
    fig, ax1 = plt.subplots(tight_layout=True)
    ax1.hist(data, bins=n_bins)
    plt.yscale('log')
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(output)
    plt.clf()
    end = time.time()
    logger.info(f"Plotted to {output} in {end - start} seconds")

def plot_scatter(x, y, title, xlabel, ylabel, output):
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.savefig(output)


def plot_scatter_min_max(x, y, title, xlabel, ylabel, output, xmin=None, xmax=None, ymin=None, ymax=None):
    plt.figure(figsize=(12, 8))
    # plt.ylim(top=ymax)
    plt.scatter(x, y, s=0.5)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.savefig(output)


from importlib import import_module
import sys
import argparse
import os
from pathlib import PurePath, Path

plot_import = import_module("sample-plot")
plot_main = getattr(plot_import, "main")

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample-count", "-s", dest="samples", type=int, required=True)
    parser.add_argument("--base-directory", "-d", dest="directory", required=True)
    parser.add_argument("--policy", dest="policy", required=True)
    parser.add_argument("--average", "--avg", "-a", dest="avg", action="store_true", default=False)
    parser.add_argument("--skip-missing", action="store_true", help="Whether to skip graphing missing CSV files.")

    # parser.add_argument("--dont-show", dest="show", action="store_false", default=True, help="Whether to not show the plot and only write to disk.")
    options = parser.parse_args(args)
    cmd = ""
    for mem in ["0.5", "0.75", "0.85", "0.95", "1.0"]:
        cmd = f"-d {options.directory} --policy {options.policy} --memory {mem} -s {options.samples} --dont-show"
        if options.avg:
            cmd += " --average"
        if options.skip_missing:
            cmd += " --skip-missing"
        command = cmd.split(" ")

        plot_main(command)


if __name__ == "__main__":
    main()
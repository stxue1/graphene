import argparse
import shutil
import subprocess
import sys
import os


def main(args=None):
    args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-directory", "-d", dest="directory", required=True)
    parser.add_argument("--sample-count", "-s", dest="samples", type=int, required=True)
    options = parser.parse_args(args)
    """
python percentile.py --base-directory Alibaba/csv_dag_fifo/dag_updated_vms_0.00001_1 -s 100 --dont-show --memory 0.5 --output percentile_0.00001_mem_0.5.pdf
    """
    memory = [0.5, 0.75, 0.85, 0.95]
    fifo = any("fifo" in folder for folder in options.directory.split("/"))
    for mem in memory:
        probability = os.path.basename(options.directory).split("_")[-2]
        output_filename = f"percentile_{probability}_mem_{mem}.pdf"
        cmd = f"python percentile.py --base-directory {options.directory} -s {options.samples} --dont-show --memory {mem} --output {output_filename}"
        p = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            p.check_returncode()
        except subprocess.CalledProcessError as e:
            print(e.output)
            print(e.stdout)
            print(e.stderr)
            raise e
        alibaba_folder = f"Alibaba/dag_{'FIFO' if fifo else 'SJF'}/{probability}"
        shutil.move(output_filename, os.path.join(alibaba_folder, output_filename))

if __name__ == "__main__":
    main()
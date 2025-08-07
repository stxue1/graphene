import shutil


def main():
    # deal with 0.001 first
    probability = "0.001"
    for i in range(1, 11):
        file = f"Alibaba/trace-sample/updated_vms_{probability}_{i}.csv"
        new_file = f"Alibaba/trace-sample/dag_updated_vms_{probability}_{i}.csv"
        shutil.move(file, new_file)

    #0.0001
    probability = "0.0001"
    for i in range(1, 11):
        file = f"Alibaba/trace-sample/updated_vms_{probability}_{i}.csv"
        new_file = f"Alibaba/trace-sample/dag_updated_vms_{probability}_{i}.csv"
        shutil.move(file, new_file)

    probability = "0.00001"
    for i in range(1, 101):
        file = f"Alibaba/trace-sample/updated_vms_{probability}_{i}.csv"
        new_file = f"Alibaba/trace-sample/dag_updated_vms_{probability}_{i}.csv"
        shutil.move(file, new_file)


if __name__ == "__main__":
    main()

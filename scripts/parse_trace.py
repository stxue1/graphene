import math

import pandas as pd


def main():
    file_path = 'alibaba-2018.csv'
    df = pd.read_csv(file_path, names=['task_name', 'instance_num', 'job_name', 'task_type', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem'])

    terminated_tasks = df[df['status'] == 'Terminated'].copy()
    # total_instances = terminated_tasks['instance_num'].sum()
    # print(f"Total number of instances in the trace: {total_instances}")

    # terminated_tasks.dropna(subset=["plan_cpu", "plan_mem"], inplace=True)
    # df_np = terminated_tasks.to_numpy()

    # i = 0
    # for row in df_np:
    #     task_name = row[0]
    #     instance_num = row[1]
    #     job_name = row[2]
    #     task_type = row[3]
    #     status = row[4]
    #     start_time = row[5]
    #     end_time = row[6]
    #     plan_cpu = row[7]
    #     plan_mem = row[8]
    #     try:
    #         a = math.ceil(plan_cpu / 100)
    #     except:
    #         print(plan_cpu, i)
    #         raise
    #     i += 1
    terminated_tasks['Core'] = terminated_tasks['plan_cpu'] / 100  # don't get fractional cpus
    terminated_tasks['Memory'] = terminated_tasks['plan_mem'] / 100  # normalize to 1
    terminated_tasks['Lifetime'] = terminated_tasks['end_time'] - terminated_tasks['start_time']
    terminated_tasks = terminated_tasks[terminated_tasks['Lifetime'] > 0].copy()

    result = terminated_tasks[['start_time', 'end_time', 'Core', 'Memory', 'instance_num', 'Lifetime', 'job_name', 'task_name']]
    result.columns = ['Start Time', 'End Time', 'Core', 'Memory', 'Instance Num', 'Lifetime', 'Job Name', 'Task Name']

    result = result.sort_values('Start Time')

    output_file = 'updated_vms.csv'
    result.to_csv(output_file, index=False)

    print(f"Processed data saved to {output_file}")


if __name__ == "__main__":
    main()

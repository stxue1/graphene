import scheduler

def test_twork():
    """
    This also tests t_work
    """
    resources = ["mem", "cpu"]
    dag_allocation = {"cpu": 4, "mem": 4}
    task_durations = [1, 1, 1, 1]
    task_demands = {"cpu": [1, 1, 1, 1], "mem": [1, 1, 1, 1]}
    exec_time = 1
    twork = scheduler.calculate_fragscore(resources, dag_allocation, task_durations, task_demands, exec_time)
    assert twork == 1.0

    dag_allocation = {"cpu": 4, "mem": 4}
    task_durations = [1, 2, 3, 4]
    task_demands = {"cpu": [1, 1, 1, 1], "mem": [1, 1, 1, 1]}
    exec_time = 1
    twork = scheduler.calculate_fragscore(resources, dag_allocation, task_durations, task_demands, exec_time)
    # essentially, this should mean it takes 2.5 seconds of the cluster share resources to complete
    assert twork == 2.5

def test_fragscore():

    resources = ["mem", "cpu"]
    dag_allocation = {"cpu": 6 , "mem": 6}
    task_durations = [2, 2, 2, 2]
    task_demands = {"cpu": [1, 1, 1, 1], "mem": [1, 1, 1, 1]}
    exec_time = 4
    fragscore = scheduler.calculate_fragscore(resources, dag_allocation, task_durations, task_demands, exec_time)
    assert fragscore == 1/3

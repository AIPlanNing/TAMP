import os


def check_task(data_file, task):
    _, train_file = os.path.split(data_file)
    prefix = train_file.split('_')[0]
    if prefix == task:
        return True
    raise Warning(f'task {task} does not match data file {data_file}')
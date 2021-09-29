#@Time      :2019/12/23 19:48
#@Author    :zhounan
#@FileName  :cmd_attack_0.py
import numpy as np
from multiprocessing import Pool
from itertools import product
import pynvml
import random

def temp_run(args):
    import time,os,sched
    schedule = sched.scheduler(time.time,time.sleep)
    def perform_command(cmd,inc):
        print(cmd)
        os.system(cmd)
        print()
        print('task')
    def timming_exe(cmd,inc=60):
      schedule.enter(inc,0,perform_command,(cmd,inc))
      schedule.run()
    pynvml.nvmlInit()
    while True:
        time.sleep(args[0] * 4)
        for i in gpu_list:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if len(info) == 0:
                timming_exe(
                    "CUDA_VISIBLE_DEVICES={} python train_{}.py --n_epochs=40 --path_suffix='{}' --train_type=loss_ensemble --loss_type={} --dynamic_type=normal --alpha={} --para_flag={}"
                    .format(i, args[1], args[2], args[3], args[4], args[5]),
                    1)
                exit(0)

def temp_run1(args):
    import time,os,sched
    schedule = sched.scheduler(time.time,time.sleep)
    def perform_command(cmd,inc):
        print(cmd)
        os.system(cmd)
        print()
        print('task')
    def timming_exe(cmd,inc=60):
      schedule.enter(inc,0,perform_command,(cmd,inc))
      schedule.run()

    timming_exe(
        "CUDA_VISIBLE_DEVICES={} python train_{}.py --n_epochs=40 --path_suffix='{}' --train_type=loss_ensemble --loss_type={} --dynamic_type=normal --alpha={} --para_flag={}"
            .format(args[0], args[1], args[2], args[3], args[4], args[5]),
        1)

# begin 20201209
# cifar10, resnet32, gal_regular_loss_new14, 未跑
def run():
    gpu_list = [0]
    gpu_num = len(gpu_list)
    data = ['cifar10']
    model = ['resnet20']
    #model = ['resnet20']
    #loss_type = ['gal_regular_loss_new12', 'gal_regular_loss_new13', 'gal_regular_loss_new14']
    loss_type = ['gal_regular_loss_new15']
    alpha = [0.01]
    args = list(product(data, model, loss_type, alpha))

    gpu_index = [gpu_list[i % gpu_num] for i in range(len(args))]
    args = [[gpu_index[i]] + list(a) + [0] for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run1():
    gpu_list = [0, 1, 2, 3]
    gpu_num = len(gpu_list)
    data = ['mnist', 'cifar10']
    #model = ['resnet26', 'resnet32']
    model = ['resnet26']
    loss_type = ['gal_regular_loss']
    # alpha = [0.025]
    alpha = [0.015]
    args = list(product(data, model, loss_type, alpha))

    gpu_index = [gpu_list[i % gpu_num] for i in range(len(args))]
    args = [[gpu_index[i]] + list(a) + [1] for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run2():
    gpu_list = list(range(0,6))
    gpu_num = len(gpu_list)
    data = ['mnist', 'cifar10']
    # model = ['resnet20', 'resnet26', 'resnet32', 'resnetmix']
    model = ['resnet20']
    loss_type = ['ce', 'gal_loss']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))

    gpu_index = [gpu_list[i % gpu_num] for i in range(len(args))]
    args = [[gpu_index[i]] + list(a) + [0] for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run3():
    gpu_list = [1, 2, 3, 4, 5]
    gpu_num = len(gpu_list)
    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    alpha = [0.025]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [1] for a in args]


    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new15']
    alpha = [0.01]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [0] for a in args]

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss_new14']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args3 = [list(a) + [0] for a in args]

    args = args1 + args2 + args3
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run4():
    gpu_list = [1, 2, 3, 4, 5]
    gpu_num = len(gpu_list)
    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    alpha = [0.015]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [1] for a in args]


    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new15']
    alpha = [0.01]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [0] for a in args]


    args = args1 + args2
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run5():
    gpu_list = [0, 1, 3, 5]
    gpu_num = len(gpu_list)
    data = ['mnist', 'cifar10']
    #model = ['resnet26', 'resnet32']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    # alpha = [0.025]
    alpha = [0.018]
    args = list(product(data, model, loss_type, alpha))
    args = [list(a) + [1] for a in args]

    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run6():
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['mnist']
    model = ['resnet20', 'resnet26', 'resnet32', 'resnetmix']
    loss_type = ['gal_regular_loss']
    alpha = [0.015]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet32', 'resnetmix']
    loss_type = ['gal_regular_loss']
    alpha = [0.015]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [1] for a in args]

    args = args1 + args2
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run7():
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['mnist']
    model = ['resnet20']
    loss_type = ['gal_regular_loss_new15']
    alpha = [0.01]
    args = list(product(data, model, loss_type, alpha))
    args = [list(a) + [1] for a in args]

    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args)

def run8():
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['mnist', 'cifar10']
    model = ['resnet20', 'resnet26', 'resnet32', 'resnetmix']
    loss_type = ['ce', 'gal_loss']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [0] for a in args]

    data = ['mnist', 'cifar10']
    model = ['resnet20', 'resnet26', 'resnet32', 'resnetmix']
    loss_type = ['gal_regular_loss']
    alpha = [0.01]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    alpha = [0.018]
    args = list(product(data, model, loss_type, alpha))
    args3 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new15']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args4 = [list(a) + [0] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new_iter']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args5 = [list(a) + [0] for a in args]

    args = args1 + args2 + args3 + args4 + args5
    args = [[gpu_list] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run9():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['mnist']
    model = ['resnet32']
    loss_type = ['gal_loss']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [0] for a in args]

    data = ['mnist', 'cifar10']
    model = ['resnet26', 'resnet32', 'resnetmix']
    loss_type = ['gal_regular_loss']
    alpha = [0.01]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    alpha = [0.018]
    args = list(product(data, model, loss_type, alpha))
    args3 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new15']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args4 = [list(a) + [0] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new_iter']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args5 = [list(a) + [0] for a in args]

    args = args1 + args2 + args3 + args4 + args5
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run10():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new_iter']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args5 = [list(a) + [0] for a in args]

    pynvml.nvmlInit()
    while True:
        free_num = 0
        for i in gpu_list:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if len(info) == 0:
                free_num = free_num + 1
        if free_num >= 2:
            break
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run11():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['mnist', 'cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    alpha = [0.005, 0.015]
    args = list(product(data, model, loss_type, alpha))
    args = [list(a) + [1] for a in args]

    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run12():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['mnist', 'cifar10']
    model = ['resnet20', 'resnet26', 'resnet32', 'resnetmix']
    loss_type = ['gal_loss']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [0] for a in args]

    data = ['mnist', 'cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    alpha = [0.01]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [1] for a in args]

    # data = ['mnist', 'cifar10']
    # model = ['resnet20', 'resnet26', 'resnet32', 'resnetmix']
    # loss_type = ['gal_regular_loss']
    # alpha = [0.01]
    # args = list(product(data, model, loss_type, alpha))
    # args2 = [list(a) + [1] for a in args]

    # data = ['cifar10']
    # model = ['resnet20', 'resnet26']
    # loss_type = ['gal_regular_loss']
    # alpha = [0.018]
    # args = list(product(data, model, loss_type, alpha))
    # args3 = [list(a) + [1] for a in args]
    #
    # data = ['cifar10']
    # model = ['resnet20', 'resnet26']
    # loss_type = ['gal_regular_loss_new15']
    # alpha = [0]
    # args = list(product(data, model, loss_type, alpha))
    # args4 = [list(a) + [0] for a in args]
    #
    # data = ['cifar10']
    # model = ['resnet20', 'resnet26']
    # loss_type = ['gal_regular_loss_new_iter']
    # alpha = [0]
    # args = list(product(data, model, loss_type, alpha))
    # args5 = [list(a) + [0] for a in args]

    #args = args1 + args2 + args3 + args4 + args5
    args = args1 + args2
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run13():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss']
    alpha = [0.5, 0.1, 0.05, 0.005, 0.001]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss']
    alpha = [0.018]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20', 'resnet26']
    loss_type = ['gal_regular_loss_new_iter']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args3 = [list(a) + [0] for a in args]

    args = args1 + args2 + args3
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(gpu_num, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run14():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss']
    alpha = [0.025, 0.075]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss_new16']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [0] for a in args]

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss']
    alpha = [10, 1]
    args = list(product(data, model, loss_type, alpha))
    args3 = [list(a) + [1] for a in args]

    args = args1 + args2 + args3
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(4, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run15():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss']
    alpha = [0.055, 0.06, 0.065, 0.07]
    args = list(product(data, model, loss_type, alpha))
    args1 = [list(a) + [1] for a in args]

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss_new17']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args2 = [list(a) + [0] for a in args]

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss_new_iter1']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args3 = [list(a) + [0] for a in args]

    args = args1 + args2 + args3
    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    with Pool(min(6, len(args))) as p:
        p.map(temp_run, args, chunksize=1)

def run16():
    global gpu_list
    gpu_list = list(range(0, 6))
    gpu_num = len(gpu_list)

    data = ['cifar10']
    model = ['resnet20']
    loss_type = ['gal_regular_loss_new17']
    alpha = [0]
    args = list(product(data, model, loss_type, alpha))
    args = [list(a) + [0] for a in args]

    args = [[gpu_list[i % gpu_num]] + a for i, a in enumerate(args)]
    temp_run(args[0])

if __name__ == '__main__':
    run16()

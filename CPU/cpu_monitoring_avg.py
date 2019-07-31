#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import psutil
import threading
import subprocess
import numpy as np
from lib.HoonUtils import setup_logger
import os

logger = setup_logger("cpu_logger", "cpu_monitoring")


def use_list_file(list_path):

    with open(list_path, 'r') as f:
        process_list = f.read().splitlines()

    pid_num_list = []
    for processes in process_list:
        grep_pid = "ps -ef | grep " + str(processes) + " | grep -v auto | grep -v grep"
        try:
            grep_pid = subprocess.check_output(grep_pid, shell=True)
        except Exception as e:
            print(e)
            continue

        grep_pid = grep_pid.decode()
        grep_pid = grep_pid.split('\n')
        for contents in grep_pid:
            content_list = ' '.join(contents.split()).split(' ')
            if len(content_list) != 1:
                pid_num = content_list[1]
                pid_num_list.append([processes, pid_num])

    return pid_num_list


def get_(size):

    power = 2**10  #2**10 = 1024
    n = 0
    Dic_powerN = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, Dic_powerN[n]+'B'


def display_progressbar(value, end_value, bar_length=40):
    percent = float(value) / end_value
    arrow = '■' * int(round(percent*bar_length)-1)
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent : [{0}] {1}%\n".format(arrow + spaces, int(round(percent*100))))
    sys.stdout.flush()


def updateit(list_path):

    pid_num_list = use_list_file(list_path)

    info_list = []

    total_cpu = 0.
    total_mem = 0.
    total_size = 0.
    total_vsize = 0.

    for j in pid_num_list:
        check_command_pid = "ps -o pid -p " + str(j[1])
        # check_command_psr = "ps -o psr -p " + str(j[1])
        # check_command_comm = "ps -o comm -p " + str(j[1])
        check_command_pcpu = "ps -o pcpu -p " + str(j[1])
        check_command_pmem = "ps -o pmem -p " + str(j[1])
        check_command_size = "ps -o size -p " + str(j[1])
        check_command_vsize = "ps -o vsize -p " + str(j[1])
        process_name = str(j[0])

        try:
            pid_num = subprocess.check_output(check_command_pid, shell=True)
            pid_num = pid_num.decode()
            pid_num = pid_num.split('\n')[1].split(' ')[-1]

            pid_cpu_usage = subprocess.check_output(check_command_pcpu, shell=True)
            pid_cpu_usage = pid_cpu_usage.decode()
            pid_cpu_usage = pid_cpu_usage.split('\n')[1].split(' ')[-1]

            total_cpu += float(pid_cpu_usage)

            pid_mem = subprocess.check_output(check_command_pmem, shell=True)
            pid_mem = pid_mem.decode()
            pid_mem = pid_mem.split('\n')[1].split(' ')[-1]

            total_mem += float(pid_mem)

            pid_size = subprocess.check_output(check_command_size, shell=True)
            pid_size = pid_size.decode()
            pid_size = pid_size.split('\n')[1].split(' ')[-1]

            total_size += float(pid_size)

            pid_vsize = subprocess.check_output(check_command_vsize, shell=True)
            pid_vsize = pid_vsize.decode()
            pid_vsize = pid_vsize.split('\n')[1].split(' ')[-1]

            total_vsize += float(pid_vsize)

            info_list.append([pid_num, process_name, float(pid_cpu_usage), float(pid_mem), float(pid_size),
                              float(pid_vsize), float(total_cpu), float(total_mem), float(total_size), float(total_vsize)])

        except Exception as e:
            print(e)
            continue

    return info_list


def show_result():
    cpu_count = psutil.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    ram_usage = psutil.virtual_memory().percent

    process_info_all = '\n'
    process_info_sum = '\n'

    if len(sys.argv) >= 3:
        list_path = sys.argv[2]
    else:
        list_path = "./cpu_monitoring_list.txt"

    info_list = updateit(list_path)

    for info in info_list:
        process_name = info[1]
        process_id = info[0]
        process_cpu_usage = info[2]
        process_mem_usage = info[3]
        process_size = info[4]
        process_vsize = info[5]
        total_cpu = info[6]
        total_mem = info[7]
        total_size = info[8]
        total_vsize = info[9]

        conv_process_size = get_(process_size * 1024)
        conv_process_vsize = get_(process_vsize * 1024)
        conv_total_size = get_(total_size * 1024)
        conv_total_vsize = get_(total_vsize * 1024)

        process_info = " {0: <15s} (PID : {1:5s}) || CPU Usage = {2: 7.1f}% || MEM Usage = {3: 6.1f}% || " \
                       "Size = {4: 4.1f}{5:2s} || Virtual Size = {6: 4.1f}{7:2s} \n".format(str(process_name),
                                                                                            process_id,
                                                                                            process_cpu_usage,
                                                                                            process_mem_usage,
                                                                                            conv_process_size[0],
                                                                                            conv_process_size[1],
                                                                                            conv_process_vsize[0],
                                                                                            conv_process_vsize[1])
        process_info_all += process_info

    process_info_sum_format = "CPU Usage = {: 4.1f}% || MEM Usage = {: 4.1f}% || Size = {: 4.1f}{:2s} || " \
                              "Virtual Size = {: 4.1f}{:2s} \n".format(total_cpu, total_mem, conv_total_size[0],
                                                                       conv_total_size[1], conv_total_vsize[0],
                                                                       conv_total_vsize[1])

    process_info_sum += process_info_sum_format

    all_cpu_usage = []
    for i in range(cpu_count):
        per_cpu_usage = cpu_usage[i]
        all_cpu_usage.append(per_cpu_usage)

    all_cpu_usage_array = np.array(all_cpu_usage)
    all_cpu_usage_avg = np.mean(all_cpu_usage_array)

    logger.info(" {0:<20s} {1:4.2f}%".format('CPU Usage average : ', round(all_cpu_usage_avg, 2)))
    logger.info(" {0:<20s} {1:4.2f}%".format('RAM usage : ', ram_usage))
    logger.info("\n----------------------------------")
    logger.info("\n[Per Process Usage Info]")
    logger.info(process_info_all)
    logger.info(process_info_sum)

    threading.Timer(int(sys.argv[1]), show_result).start()


if __name__ == '__main__':
    interval = sys.argv[1]
    show_result()

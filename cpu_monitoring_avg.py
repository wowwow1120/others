#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import psutil
import threading
import subprocess
import configparser
import numpy as np
import json
from lib.HoonUtils import setup_logger




def use_ini_file():
    config = configparser.ConfigParser()
    config.read('cpu_monitoring.ini')
    process_list = []
    for _, value in config['process_list'].items():
        process = value
        process_list.append(process)

    pid_num_list = []
    for processes in process_list:
        grep_pid = "ps -ef | grep " + str(processes) + " | grep -v auto | grep -v grep"
        grep_pid = subprocess.check_output(grep_pid, shell=True)
        grep_pid = grep_pid.decode()
        grep_pid = grep_pid.split('\n')
        for contents in grep_pid:
            content_list = ' '.join(contents.split()).split(' ')
            if len(content_list) != 1:
                pid_num = content_list[1]
                pid_num_list.append(pid_num)

    return pid_num_list


def display_progressbar(value, end_value, bar_length=40):
    percent = float(value) / end_value
    arrow = '■' * int(round(percent*bar_length)-1)
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent : [{0}] {1}%\n".format(arrow + spaces, int(round(percent*100))))
    sys.stdout.flush()


def updateit():

    logger = setup_logger("cpu_logger", "cpu_monitoring")
    threading.Timer(int(sys.argv[1]), updateit).start()
    cpu_count = psutil.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    ram_usage = psutil.virtual_memory().percent

    pid_num_list = use_ini_file()



    process_usage_sum = '\n'
    all_cpu_usage = []
    for i in range(cpu_count):
        pid_info_list = []

        for j in pid_num_list:
            check_command_pid = "ps -o pid -p " + str(j)
            check_command_psr = "ps -o psr -p " + str(j)
            check_command_comm = "ps -o comm -p " + str(j)
            # check_command_pcpu = "ps -o pcpu -p " + str(j)

            pid_num = subprocess.check_output(check_command_pid, shell=True)
            pid_num = pid_num.decode()
            pid_num = pid_num.split('\n')[1]
            pid_num = pid_num.split(' ')[-1]

            pid_cpu = subprocess.check_output(check_command_psr, shell=True)
            pid_cpu = pid_cpu.decode()
            pid_cpu = pid_cpu.split('\n')[1]
            pid_cpu = pid_cpu.split(' ')[-1]

            pid_process_name = subprocess.check_output(check_command_comm, shell=True)
            pid_process_name = pid_process_name.decode()
            pid_process_name = pid_process_name.split('\n')[1]
            pid_process_name = pid_process_name.split(' ')[-1]



            if str(pid_cpu) == str(i):
                x = [pid_num, pid_process_name]
                pid_info_list.append(x)

        per_cpu_usage = cpu_usage[i]

        if len(pid_info_list) != 0:
            for process in pid_info_list:
                process_name = process[1]
                process_id = process[0]
                process_usage = " {0: <15s}(PID : {1:5s}) : CPU {2:2d} || Usage = {3: 4.1f}% \n".format(str(process_name), process_id, i, per_cpu_usage)
                process_usage_sum += process_usage


        all_cpu_usage.append(per_cpu_usage)

    all_cpu_usage_array = np.array(all_cpu_usage)
    all_cpu_usage_avg = np.mean(all_cpu_usage_array)
    # print(" {0:<20s} {1:4.2f}%".format('CPU Usage average : ', round(all_cpu_usage_avg, 2)))
    # print(" {0:<20s} {1:4.2f}%".format('RAM usage : ', ram_usage))
    #
    #
    # print('\n', "----------------------------------")
    # print('\n', "[Per Process Usage Info]")
    # print(process_usage_sum)

    logger.info(" {0:<20s} {1:4.2f}%".format('CPU Usage average : ', round(all_cpu_usage_avg, 2)))
    logger.info(" {0:<20s} {1:4.2f}%".format('RAM usage : ', ram_usage))
    logger.info('\n', "----------------------------------")
    logger.info('\n', "[Per Process Usage Info]")
    logger.info(process_usage_sum)




if __name__ == '__main__':
    interval = sys.argv[1]

    updateit()

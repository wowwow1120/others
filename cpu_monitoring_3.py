#! /usr/bin/python
# -*- coding: utf-8 -*-

import sys
import psutil
import threading
import subprocess
import configparser
import time


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
    #time.sleep(5)
    threading.Timer(int(sys.argv[1]), updateit).start()
    cpu_count = psutil.cpu_count()
    # cpu_stats = psutil.cpu_stats()
    # cpu_freq = psutil.cpu_freq()
    # cpu_load = psutil.getloadavg()
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    ram_usage = psutil.virtual_memory().percent

    pid_num_list = use_ini_file()

    print('\n', "CPU_count = ", cpu_count)
    # print('\n', "[CPU_status]", '\n', "ctx_switches = ",cpu_stats[0],'\n',
    #       "interrupts = ",cpu_stats[1],'\n',"soft_interrupts = ",cpu_stats[2],'\n', "syscalls = ",cpu_stats[3])
    # print('\n', "[CPU_freq]", '\n', "current = ",cpu_freq[0], '\n', "min = ",cpu_freq[1],'\n', "max = ",cpu_freq[2])
    # print('\n', "[CPU_load]", '\n', "load = ",cpu_load)

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

            # pid_usage = subprocess.check_output(check_command_pcpu, shell=True)
            # pid_usage = pid_usage.decode()
            # pid_usage = pid_usage.split('\n')[1]
            # pid_usage = pid_usage.split(' ')[-1]

            if str(pid_cpu) == str(i):
                x = [pid_num, pid_process_name]
                pid_info_list.append(x)

        per_cpu_usage = cpu_usage[i]

        print("CPU", i, " : ", pid_info_list, per_cpu_usage)
        # print("CPU", cpu_num, " : ", pid_list)

        # cpu_usage_bar_value = str(per_cpu_usage).split(',')[0]
        # cpu_usage_bar_value = cpu_usage_bar_value.split('(')[1]
        # cpu_usage_bar_value = cpu_usage_bar_value.split('=')[1]
        # # print('\n', "CPU", cpu_num, " : ", i)
        # print(display_progressbar(cpu_usage_bar_value, 100))
        # cpu_num += 1

    print('\n', "--------------")

    print('\n', "RAM_usage : ", ram_usage, '\n')


if __name__ == '__main__':
    interval = sys.argv[1]

    updateit()

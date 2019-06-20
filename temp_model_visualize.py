import pandas as pd
from scipy import stats
from numpy import mean
import matplotlib.pyplot as plt

read_norm_dataset = pd.read_csv("C:\PycharmProjects\posco-temp-model\sys_temp\\analysis\\norm_range_data.csv")
read_abnorm_dataset = pd.read_csv("C:\PycharmProjects\posco-temp-model\sys_temp\\analysis\\abnorm_range_data.csv")


#norm_data 와 abnorm_data의 통계치
summary_norm_dataset = read_norm_dataset.describe()
summary_abnorm_dataset = read_abnorm_dataset.describe()
print(summary_norm_dataset)
print(summary_abnorm_dataset)

#norm_data 와 abnorm_data의 검정통계량과 p-value


column_norm_data_list = []
for i in read_norm_dataset['CONV_ENDPNT_TEMP']:
    column_norm_data_list.append(i)

print(column_norm_data_list)

column_abnorm_data_list = []
for i in read_abnorm_dataset['CONV_ENDPNT_TEMP']:
    column_abnorm_data_list.append(i)

print(column_abnorm_data_list)

chis = stats.chisquare(column_norm_data_list, column_abnorm_data_list)
print(chis)

# list = []
# for i in range(1550, 1800):
#     list.append(i)
#
# TEMP_output_hist = plt.hist(read_norm_dataset['CONV_ENDPNT_TEMP'], list)
# TEMP_output_hist = plt.xlabel('CONV_ENDPNT_TEMP')
# TEMP_output_hist = plt.ylabel('Frequency')
# TEMP_output_hist = plt.title('Histogram of CONV_ENDPNT_TEMP')
# #TEMP_output_hist = plt.gcf()
# #TEMP_output_hist = plt.show()
# TEMP_output_hist = plt.savefig('C:\PycharmProjects\\CONV_ENDPNT_TEMP.png')
# TEMP_output_hist
#
#
# TEMP_output_hist = plt.hist(read_abnorm_dataset['CONV_ENDPNT_TEMP'], list)
# TEMP_output_hist = plt.xlabel('CONV_ENDPNT_TEMP')
# TEMP_output_hist = plt.ylabel('Frequency')
# TEMP_output_hist = plt.title('Histogram of CONV_ENDPNT_TEMP')
# #TEMP_output_hist = plt.gcf()
# #TEMP_output_hist = plt.show()
# TEMP_output_hist = plt.savefig('C:\PycharmProjects\\CONV_ENDPNT_TEMP_3.png')
# TEMP_output_hist

# TEMP_output_hist2 = plt.hist(read_dataset['EVAL_TEMP'], list)
# TEMP_output_hist2 = plt.xlabel('EVAL_TEMP')
# TEMP_output_hist2 = plt.ylabel('Frequency')
# TEMP_output_hist2 = plt.title('Histogram of EVAL_TEMP')
# #TEMP_output_hist = plt.gcf()
# #TEMP_output_hist = plt.show()
# TEMP_output_hist2 = plt.savefig('C:\PycharmProjects\\EVAL_TEMP_2.png')
# TEMP_output_hist2

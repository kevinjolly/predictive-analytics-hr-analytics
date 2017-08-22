import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("cleaned_df.csv")
original_df = pd.read_csv("HR_comma_sep.csv")

# PLOT 1
import seaborn as sns
sns.lmplot(x='average_montly_hours', y='satisfaction_level',
           data=df, hue='left', palette='Set1')
plt.show()

# PLOT 2
sns.pairplot(original_df, hue="left")
plt.show()

# PLOT 3
sns.jointplot(x="satisfaction_level", y="last_evaluation", data=df, kind='kde')
plt.show()

# PLOT 4
sns.jointplot(x="satisfaction_level",
              y="average_montly_hours", data=df, kind='kde')
plt.show()

# PLOT 5
sns.jointplot(x="average_montly_hours",
              y="last_evaluation", data=df, kind='kde')
plt.show()

# PLOT 6

import matplotlib.pyplot as plt
df.number_project.plot('hist')
plt.show()

# PLOT 7
df.boxplot(column='satisfaction_level', by='left')
plt.show()

# PLOT 8
df.boxplot(column='last_evaluation', by='left')
plt.show()

# PLOT 9
df.boxplot(column='average_montly_hours', by='left')
plt.show()

# PLOT 10
df.boxplot(column='time_spend_company', by='left')
plt.show()

# PLOT 11
df.boxplot(column='number_project', by='left')
plt.show()

# PLOT 12
import seaborn as sns
sns.stripplot(x='left', y='average_montly_hours', data=df)
plt.show()

# PLOT 13
sns.stripplot(x='left', y='number_project', data=df)
plt.show()

# PLOT 14
sns.stripplot(x='left', y='last_evaluation', data=df)
plt.show()

# PLOT 15:
sns.stripplot(x='left', y='satisfaction_level', data=df)
plt.show()

# PLOT 16:
df['average_montly_hours'].hist(by=df['left'])
plt.show()

# PLOT 17:
df['time_spend_company'].hist(by=df['left'])
plt.show()

# PLOT 18:
df['salary'].hist(by=df['left'])
plt.show()

# PLOT 19:
df['department'].hist(by=df['left'])
plt.show()

# PLOT 20:
df['number_project'].hist(by=df['left'])
plt.show()

# PLOT 21:
df['satisfaction_level'].hist(by=df['left'])
plt.show()

# PLOT 22:
df['last_evaluation'].hist(by=df['left'])
plt.show()

# PLOT 23:
df['work_accident'].hist(by=df['left'])
plt.show()

# PLOT 24:
df['promotion_last_5years'].hist(by=df['left'])
plt.show()

# PLOT 25:
sns.violinplot(x='left', y='average_montly_hours', data=df)
plt.show()

# PLOT 26:
sns.violinplot(x='left', y='last_evaluation', data=df)
plt.show()

# PLOT 27:
sns.violinplot(x='left', y='satisfaction_level', data=df)
plt.show()

# PLOT 28:
sns.violinplot(x='left', y='number_project', data=df)
plt.show()

# PLOT 29:
sns.violinplot(x='left', y='time_spend_company', data=df)
plt.show()

# PLOT 30:
sns.violinplot(x='salary', y='satisfaction_level', data=df)
plt.show()

# PLOT 31
sns.violinplot(x='salary', y='number_project', data=df)
plt.show()

# PLOT 32:
sns.violinplot(x='salary', y='last_evaluation', data=df)
plt.show()

# PLOT 33:
sns.violinplot(x='salary', y='average_montly_hours', data=df)
plt.show()

# PLOT 34:
sns.violinplot(x='department', y='average_montly_hours', data=df)
plt.show()

# PLOT 35:
sns.violinplot(x='department', y='satisfaction_level', data=df)
plt.show()

# PLOT 36:
sns.violinplot(x='department', y='last_evaluation', data=df)
plt.show()

# PLOT 37:
sns.violinplot(x='department', y='time_spend_company', data=df)
plt.show()

# PLOT 38:
import numpy as np
percentiles = np.array([25, 50, 75])
ptiles = np.percentile(df['satisfaction_level'], percentiles)
x = np.sort(df['satisfaction_level'])
y = np.arange(1, len(x) + 1) / len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.plot(ptiles, percentiles / 100, marker='D',
             color='red', linestyle='none')
plt.xlabel("Satisfaction Levels")
plt.ylabel("Emperical Cumulative Distribution Function")
plt.title("ECDF Plot of Average monthly working hours")
plt.margins(0.02)
plt.show()

# PLOT 39:
import numpy as np
percentiles = np.array([25, 50, 75])
ptiles = np.percentile(df['last_evaluation'], percentiles)
x = np.sort(df['last_evaluation'])
y = np.arange(1, len(x) + 1) / len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.plot(ptiles, percentiles / 100, marker='D',
             color='red', linestyle='none')
plt.xlabel("Last evaluation scores")
plt.ylabel("Emperical Cumulative Distribution Function")
plt.title("ECDF Plot of Average monthly working hours")
plt.margins(0.02)
plt.show()

# PLOT 40:
import numpy as np
percentiles = np.array([25, 50, 75])
ptiles = np.percentile(df['average_montly_hours'], percentiles)
x = np.sort(df['average_montly_hours'])
y = np.arange(1, len(x) + 1) / len(x)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.plot(ptiles, percentiles / 100, marker='D',
             color='red', linestyle='none')
plt.xlabel("Average Monthly Working Hours")
plt.ylabel("Emperical Cumulative Distribution Function")
plt.title("ECDF Plot of Average monthly working hours")
plt.margins(0.02)
plt.show()

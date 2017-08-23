#Checking the distribution of Satisfaction levels

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Function to compute the ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y

#Comparing Theoretical and Actual Distributons of Satisfaction levels
mean = np.mean(df["satisfaction_level"])
std = np.std(df["satisfaction_level"])
samples = np.random.normal(mean, std, size = 10000)
x, y = ecdf(df["satisfaction_level"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x,y,marker = '.', linestyle = 'none')
plt.xlabel("Satisfaction Levels")
plt.ylabel("CDFs")
plt.show()

#Comparing Theoretical and Actual distribution for satisfaction levels (Exponential)
import numpy as np

#Function to compute the ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y

#Comparing Theoretical and Actual Distributons of Satisfaction levels
mean = np.mean(df["satisfaction_level"])
samples = np.random.exponential(mean, size = 10000)
x, y = ecdf(df["satisfaction_level"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x,y,marker = '.', linestyle = 'none')
plt.xlabel("Satisfaction Levels")
plt.ylabel("CDFs")
plt.show()

# checking out the distribution of the `last_evaluation` column

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



mean = np.mean(df["last_evaluation"])
std = np.std(df["last_evaluation"])


# Function to compute the ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


# trying out a normal distribution
samples = np.random.normal(mean, std, size=10000)
x, y = ecdf(df["last_evaluation"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel("Last Evaluation")
plt.ylabel("CDFs")
plt.show()


# trying out an exponential distribution
samples = np.random.exponential(mean, size = 10000)
x, y = ecdf(df["last_evaluation"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x,y,marker = '.', linestyle = 'none')
plt.xlabel("Last Evaluation")
plt.ylabel("CDFs")
plt.show()

# checking out the distribution of the `average_monthly_hours` column

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("cleaned_df.csv")
mean = np.mean(df["average_monthly_hours"])
std = np.std(df["average_monthly_hours"])


# Function to compute the ECDF
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n + 1) / n
    return x, y


# trying out a normal distribution
samples = np.random.normal(mean, std, size=10000)
x, y = ecdf(df["average_monthly_hours"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel("Average Monthly Hours")
plt.ylabel("CDFs")
plt.show()


# trying out an exponential distribution
samples = np.random.exponential(mean, size=10000)
x, y = ecdf(df["average_monthly_hours"])
x_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel("Average Monthly Hours")
plt.ylabel("CDFs")
plt.show()

#Bootstrapping parameters 

#Defining bootstrap function

def bootstrap_replicates(data, func): 
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)
#Bootstrappig all parameters for satisfaction levels

bs_replicates_mean = np.empty(100000)
bs_replicates_var = np.empty(100000)
bs_replicates_std = np.empty(100000)
bs_replicates_median = np.empty(100000)


for i in range(100000):
    bs_replicates_mean[i] = bootstrap_replicates(
        df["satisfaction_level"], np.mean)
    bs_replicates_var[i] = bootstrap_replicates(
        df["satisfaction_level"], np.var)
    bs_replicates_std[i] = bootstrap_replicates(
        df["satisfaction_level"], np.std)
    bs_replicates_median[i] = bootstrap_replicates(
        df["satisfaction_level"], np.median)


plt.hist(bs_replicates_mean, normed=True)
plt.xlabel("Mean Satisfaction levels")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_var, normed=True)
plt.xlabel("Variance of Satisfaction levels")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_std, normed=True)
plt.xlabel("Standard Deviation of Satisfaction levels")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_median, normed=True)
plt.xlabel("Median of Satisfaction levels")
plt.ylabel("PDF")
plt.show()

conf_int_mean = np.percentile(bs_replicates_mean, [2.5, 97.5])
print("Mean:  {:.2f} and {:.2f}".format(
    conf_int_mean[0], conf_int_mean[1]))

conf_int_var = np.percentile(bs_replicates_var, [2.5, 97.5])
print("Variance: {:.2f} and {:.2f}".format(
    conf_int_var[0], conf_int_var[1]))

conf_int_std = np.percentile(bs_replicates_std, [2.5, 97.5])
print("Standard Deviation: {:.2f} and {:.2f}".format(
    conf_int_std[0], conf_int_std[1]))

conf_int_median = np.percentile(bs_replicates_median, [2.5, 97.5])
print("Median: {:.2f} and {:.2f}".format(
    conf_int_median[0], conf_int_median[1]))

#Bootstrapping all parameters for Last Evaluation:
bs_replicates_mean = np.empty(100000)
bs_replicates_var = np.empty(100000)
bs_replicates_std = np.empty(100000)
bs_replicates_median = np.empty(100000)


for i in range(100000):
    bs_replicates_mean[i] = bootstrap_replicates(
        df["last_evaluation"], np.mean)
    bs_replicates_var[i] = bootstrap_replicates(
        df["last_evaluation"], np.var)
    bs_replicates_std[i] = bootstrap_replicates(
        df["last_evaluation"], np.std)
    bs_replicates_median[i] = bootstrap_replicates(
        df["last_evaluation"], np.median)


plt.hist(bs_replicates_mean, normed=True)
plt.xlabel("Mean Last Evaluation")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_var, normed=True)
plt.xlabel("Variance of Last Evaluation")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_std, normed=True)
plt.xlabel("Standard Deviation of Last Evaluation")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_median, normed=True)
plt.xlabel("Median of Last Evaluation")
plt.ylabel("PDF")
plt.show()

conf_int_mean = np.percentile(bs_replicates_mean, [2.5, 97.5])
print("Mean:  {:.2f} and {:.2f}".format(
    conf_int_mean[0], conf_int_mean[1]))

conf_int_var = np.percentile(bs_replicates_var, [2.5, 97.5])
print("Variance: {:.2f} and {:.2f}".format(
    conf_int_var[0], conf_int_var[1]))

conf_int_std = np.percentile(bs_replicates_std, [2.5, 97.5])
print("Standard Deviation: {:.2f} and {:.2f}".format(
    conf_int_std[0], conf_int_std[1]))

conf_int_median = np.percentile(bs_replicates_median, [2.5, 97.5])
print("Median: {:.2f} and {:.2f}".format(
    conf_int_median[0], conf_int_median[1]))

#Bootstrapping all parameters for average monthly hours
bs_replicates_mean = np.empty(100000)
bs_replicates_var = np.empty(100000)
bs_replicates_std = np.empty(100000)
bs_replicates_median = np.empty(100000)


for i in range(100000):
    bs_replicates_mean[i] = bootstrap_replicates(
        df["average_monthly_hours"], np.mean)
    bs_replicates_var[i] = bootstrap_replicates(
        df["average_monthly_hours"], np.var)
    bs_replicates_std[i] = bootstrap_replicates(
        df["average_monthly_hours"], np.std)
    bs_replicates_median[i] = bootstrap_replicates(
        df["average_monthly_hours"], np.median)


plt.hist(bs_replicates_mean, normed=True)
plt.xlabel("Mean Average Monthly Hours")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_var, normed=True)
plt.xlabel("Variance of Average Monthly Hours")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_std, normed=True)
plt.xlabel("Standard Deviation of Average Monthly Hours")
plt.ylabel("PDF")
plt.show()

plt.hist(bs_replicates_median, normed=True)
plt.xlabel("Median of Average Monthly Hours")
plt.ylabel("PDF")
plt.show()

conf_int_mean = np.percentile(bs_replicates_mean, [2.5, 97.5])
print("Mean:  {:.2f} and {:.2f}".format(
    conf_int_mean[0], conf_int_mean[1]))

conf_int_var = np.percentile(bs_replicates_var, [2.5, 97.5])
print("Variance: {:.2f} and {:.2f}".format(
    conf_int_var[0], conf_int_var[1]))

conf_int_std = np.percentile(bs_replicates_std, [2.5, 97.5])
print("Standard Deviation: {:.2f} and {:.2f}".format(
    conf_int_std[0], conf_int_std[1]))

conf_int_median = np.percentile(bs_replicates_median, [2.5, 97.5])
print("Median: {:.2f} and {:.2f}".format(
    conf_int_median[0], conf_int_median[1]))


#Bootstrapping for the slope and intercept 

#Converting all interested columns to float

#Converting columns to float
df['satisfaction_level'] = df['satisfaction_level'].astype('float')
df['last_evaluation'] = df['last_evaluation'].astype('float')
df['average_montly_hours'] = df['average_montly_hours'].astype('float')
satisfaction = df['satisfaction_level']
last = df['last_evaluation']
average = df['average_montly_hours']

#Creating a function to create boostrapped replicates of slope and intercept
def draw_bs_pairs_linereg(x, y, size = 1):
    inds = np.arange(len(x))
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)
    for i in range(size):
        bs_inds = np.random.choice(inds, size = len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)
        
    return bs_slope_reps, bs_intercept_reps

#Boostrapping relationship between satisfaction levels and Last evaluation s

#For slope
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linereg(satisfaction, last, size = 10000)
print(np.percentile(bs_slope_reps, [2.5,97.5]))
plt.hist(bs_slope_reps, normed = True)
plt.xlabel("slope")
plt.ylabel("PDF")
plt.show()

#For Intercept
print(np.percentile(bs_intercept_reps, [2.5,97.5]))
plt.hist(bs_slope_reps, normed = True)
plt.xlabel("Intercept")
plt.ylabel("PDF")
plt.show()

#For lines of regression
#Plotting the bootstrapped lines of regression
x = np.array([0,100])
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i], linewidth = 0.5, alpha = 0.2, color = 'red')
    
_ = plt.plot(x = satisfaction, y = last, marker = '.', linestyle = 'none')
_ = plt.xlabel("Satisfaction Level")
_ = plt.ylabel("Last Evaluation")
plt.margins(0.02)
plt.show()

#Boostrapping relationship between Average momthly hours and last evaluation

#For slope
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linereg(average, last, size = 10000)
print(np.percentile(bs_slope_reps, [2.5,97.5]))
plt.hist(bs_slope_reps, normed = True)
plt.xlabel("slope")
plt.ylabel("PDF")
plt.show()

#For intercept
print(np.percentile(bs_intercept_reps, [2.5,97.5]))
plt.hist(bs_slope_reps, normed = True)
plt.xlabel("Intercept")
plt.ylabel("PDF")
plt.show()

#For Lines of regression
#Plotting the bootstrapped lines of regression
x = np.array([0,100])
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i], linewidth = 0.5, alpha = 0.2, color = 'red')
    
_ = plt.plot(x = average, y = last, marker = '.', linestyle = 'none')
_ = plt.xlabel("Average Monthly Hours")
_ = plt.ylabel("Last Evaluation")
plt.margins(0.02)
plt.show()

#Boostrapping relationship between average monthly hours and satisfaction levels

#For slope
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linereg(average, satisfaction, size = 10000)
print(np.percentile(bs_slope_reps, [2.5,97.5]))
plt.hist(bs_slope_reps, normed = True)
plt.xlabel("slope")
plt.ylabel("PDF")
plt.show()

#For Intercept
print(np.percentile(bs_intercept_reps, [2.5,97.5]))
plt.hist(bs_slope_reps, normed = True)
plt.xlabel("Intercept")
plt.ylabel("PDF")
plt.show()

#For Lines of regression
#Plotting the bootstrapped lines of regression
x = np.array([0,100])
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i], linewidth = 0.5, alpha = 0.2, color = 'red')
    
_ = plt.plot(x = average, y = satisfaction, marker = '.', linestyle = 'none')
_ = plt.xlabel("Average Monthly Hours")
_ = plt.ylabel("Satisfaction Levels")
plt.margins(0.02)
plt.show()

#Creating a functon to compute the pearson correlation coeff 
def pearson_r(x,y):
    corr_mat = np.corrcoef(x,y)
    return corr_mat[0,1]
#Computing the pearson correlation coeff between satisfaction levels and last evaluation

r = pearson_r(satisfaction, last)
print(r)

#Computing the pearson correlation coeff between satisfaction levels and average monthly working hours
r = pearson_r(satisfaction, average)
print(r)

#Computing the pearson correlation coeff between last evaluation and average monthly working hours
r = pearson_r(last, average)
print(r)

#Hypothesis test to see if there a weak positive correlation between last evaluation and average monthly working hours
r_obs = pearson_r(last, average)

perm_replicates = np.empty(120000)
for i in range(120000):
    last_permuted = np.random.permutation(last)
    perm_replicates[i] = pearson_r(last_permuted, average)
    
#Computing the p value
p = np.sum(perm_replicates >= r_obs) / len(perm_replicates)
print('p-val =', p)

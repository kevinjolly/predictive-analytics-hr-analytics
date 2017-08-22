import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("HR_comma_sep.csv")


# rename `Work_accident` to `work_accident`
df['work_accident'] = df['Work_accident']
df = df.drop('Work_accident', 1)

# rename `average_montly_hours` to `average_monthly_hours`
df['average_monthly_hours'] = df['average_montly_hours']
df = df.drop('average_montly_hours', 1)


# No na's found
# df.isnull().sum()


# Convert required columns to categorical
df['salary'] = df['salary'].astype('category').cat.codes
df['left'] = df['left'].astype('category').cat.codes
df['department'] = df['sales'].astype('category').cat.codes
df = df.drop('sales', 1)
df['promotion_last_5years'] = df['promotion_last_5years'].astype('category')
df['work_accident'] = df['work_accident'].astype('category')

df = df.drop_duplicates()

df.to_csv("cleaned_df.csv", index=False)

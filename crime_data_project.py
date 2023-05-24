# -*- coding: utf-8 -*-
"""

383: Project - Los Angeles Crime Data

@authors: Christian Vargas, Tiffany Andersen, Andrew Kassis, Deniz Erisgen

"""

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://data.lacity.org/resource/2nrs-mtv8.csv'
data = pd.read_csv(url)

# display the first few rows of the dataset
print(data.head())
# check the column names
print(data.columns)
# get summary statistics
print(data.describe())

# converts military time without colons to standard time
def convert_to_standard_time(military_time):
    # ensure 4-digit format
    military_time = str(military_time).zfill(4)  
    hour = int(military_time[:2])
    minute = int(military_time[2:])
    
    if hour >= 12:
        suffix = 'PM'
        if hour > 12:
            hour -= 12
    else:
        suffix = 'AM'
        if hour == 0:
            hour = 12
    
    return f'{hour:02d}:{minute:02d} {suffix}'

# convert time_occ from military time without colons to standard time
data['time_occ'] = data['time_occ'].apply(convert_to_standard_time)
print(data['time_occ'])

# feature engineering: extract relevant features related to time and location.

# extract month, day of week, and year
data['date_occ'] = pd.to_datetime(data['date_occ'])
data['month'] = data['date_occ'].dt.month
data['day_of_week'] = data['date_occ'].dt.dayofweek
data['year'] = data['date_occ'].dt.year

# Inspect column names
print(data.columns)
# Inspect the 'day_of_week' column
print(data['day_of_week'])
# Inspect the 'month' column
print(data['month'])
# Inspect the 'year' column
print(data['year'])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 22:39:00 2019

@author: rajendrareddy
"""
# Pandas Categoricals:
import pandas as pd

if __name__ == '__main__':
    # Setup
    df = pandas.DataFrame({'id':[1,2,3,4,5,6],'raw_grade':['a','b','b','a','a','e']})
    print('Original df is:')
    print(df)
    
    #Create a new cloumn as type 'category'
    df['grade'] = df['raw_grade'].astype('category')
    print(df['grade'])
    
    #Rename categories and add the missing categories:
    df['grade'] = df['grade'].cat.set_categories(['very bad','bad','medium','good','very good'])
    
    #Sort by grade:
    df = df.sort('grade')
    
    #Group by grade and size:
    
    df = df.groupby('grade').size()
    print(df)
----------------------------
import pandas as pd
if __name__ == '__main__':
    raw_data = {'name':['Miller','Jacobson','Ali','Milner','Cooze','Jacon','Ryaner','Sone','Sloan','Piger','Riani','Ali'],
                'score':[1,24,25,26,30,50,55,65,74,75,80,100]}
    df = pd.DataFrame(raw_data, columns = ['name','score'])
    
    print(df)
    
    df['score'] = df['score'].astype('category')
    print(df['score']) #categories(5,int64): [25<57<62<70<94]
    df['score'].cat.set_categories(['Low','Okay','Good','Great'])
    bins = [0, 25, 50, 75, 100]
    group_names = ['Low','Okay','Good','Great']
    
    cat_obj = pd.cut(df['score'], bins, labels=group_names)
    print(cat_obj)
    
    df['cat'] = pd.cut(df['score'], bins, labels=group_names)
    
    df = df.groupby('cat').size()
    print(df)
-------------------------------

# Pandas Lesson-1

import os
from os import remove
import pandas
import matplotlib
from pandas import DataFrame, read_csv
import matplotlib.pyplot
from ggplot import *
import ggplot

names = ['Bob','Jessica','Mary','John','Mel']
births = [986,155,77,578,973]

'''zip(seq1 [, seq2 [...]]) -> [(seq1[0], seq2[0] ...), (...)]
Return a list of tuples, where each tuple contains the i-th element
from each of the argument sequences.  The returned list is truncated
in length to the length of the shortest argument sequence.'''

BabyDataSet = list(zip(names,births))
print(BabyDataSet)

#Setup DataFrame
df = DataFrame(data=BabyDataSet, columns=['Name','Births'])
print(df)

#Saving the Reading file:
df.to_csv('Births123.csv', index=False, header=True)

print("\nData Type of Births columns is: ",df.Births.dtype)

Sorted = df.sort_values(['Births'], ascending=[1])
print("Highest births by dataframe ", Sorted.head())

print("Maximium births ", df['Births'].min())

print(ggplot(aes(x='Names', y='Births'), data=df))
-------------------------

# Pandas Lesson-2
from pandas import DataFrame, read_csv
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

# List of names
names = ['Bob','Jessica','Mary','John','Mel']

if __name__ == '__main__':
    random.seed(500) # 500 Random samples can be reproduced:
    
    #Setup random names, births, and store as DataFrame
    random_names = [names[random.randint(low=0, high=len(names))] for i in range(1000)]
    birth = [random.randint(low=0, high=1000) for i in range(1000)]
    print("Births, last 10 ", births[:10])
    
    BabyDataSet = list(zip(random_names, births))
    print("BabyDataSet, last 10 :", BabyDataSet[:10])
    
    df = DataFrame(data=BabyDataSet, columns= ['Names','Births'])
    print("DataFrame, last 10: \n", df[:10])
    
    # Get unique properties
    print("Unique Names Array", df['Names'].unique())
    for x in df['Names'].unique():
        print(x)
    
    # Describe a dataframe
    print(df['Names'].describe())
    
    # Use groupby function to aggregate data
    Name = df.groupby('Names')
    df = Name.sum()
    print(df)
    
    # To get highest occurring names
    Sorted = df.sort_values(['Births'], ascending=[0])
    print(Sorted.head(1))
    
    # To get highest occuring names
    print(df['Births'].max())
    
    # Presenting Data
    df['Births'].plot(kind='hist')
    plt.show()
    
    print("Most popular name")
    df.sort_values('Births', ascending =False)
----------------------------------

# Pandas Lesson-3
import os
from openpyxl import Workbook
from pandas import DataFrame, date_range, read_excel, concat
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as np

def CreateDataSet(Number=1):
    Output = []
    for i in range(Number):
        
        #Create a Weekly (Mondays) date range:
        rng = date_range(start='1/1/2009', end='12/31/2012', freq='W-MON')
        
        #Create random data
        data = np.randint(low=25, high=1000, size=len(rng))
        
        # Status pool
        status = [1,2,3]
        
        # Make a random list of statuses:
        random_status = [status[np.randint(low=0, high=len(status))] for i in range(len(rng))]
        
        # State pool
        states = ['GA','FL','fl','NY','NJ','TX']
        
        # Make a random list of states:
        random_states = [states[np.randint(low=0, high=len(states))] for i in range(len(rng))]
        
        Output.extend(zip(random_states, random_status, data, rng))
    return Output

if __name__ == "__main__":
    #Setting up data
    np.seed(500) # To reproduce results
    dataset = CreateDataSet(4)
    df = DataFrame(data=dataset, columns=['State','Status','CustomerCount','StatusDate'])
    print(df.info())
    
    print(df.head())
    df.to_excel('Lesson-3.xlsx', index=False)
---------------------------

# Pandas Lesson-4

from pandas import DataFrame
# Setup data

d = [0,1,2,3,4,5,6,7,8,9]

#Create dataframe
df = DataFrame(d)
print(df)

# Change column name:
df.columns = ['Rev']
print(df.head(10))

#Add a column:
df['NewCol'] = 5
print(df.head(10))

#Modify values in column
df['NewCol'] = df['NewCol']+1
print(df.head())

#Deleting columns
del df['NewCol']
print(df.head())

# Create 2 new columns test and col for 10 rows and 3 cols:
df['test'] = 3
df['col'] = df['Rev']
print(df)
#print values
print(df.values)
#print column names
print(list(df.columns))
print(df.columns)

# Modifying Index
myIndex = ['a','b','c','d','e','f','g','h','i','j']
df.index = myIndex

#Using loc[inclusive:inclusive]
print(df.loc['a':'d'])
# Selecting data (iloc)
print(df.iloc[0:4])

#df['ColumnName'][inclusive:exclusive]
print (df['Rev'][0:3]) # Get column 'Rev' with rows 0, 1, 2
print (df['col'][5:]) # Get column 'col' with rows 5+
print (df[['col','test']][:3]) # Get columns 'col', 'test' with rows 0, 1, 2

print(df.head())
print(df.tail())
-----------------------------

# Lookup Lesson-4

import pandas as pd
import numpy as np

data_a = pd.DataFrame({
        'date':[
            '1959-03-01 00:00:00', '1959-03-02 00:00:00', '1959-03-03 00:00:00',
            '1959-03-31 00:00:00', '1959-06-02 00:00:00', '1959-06-04 00:00:00',
            '1959-09-30 00:00:00', '1959-10-01 00:00:00', '1959-10-15 00:00:00',
            '1959-12-31 00:00:00'],
        'item':['realgdp', 'infl', 'unemp', 'realgdp', 'infl', 'unemp',
            'realgdp', 'infl', 'unemp', 'realgdp'],
        'value':[2710.349, 0.000, 5.800, 2778.801, 2.340, 5.100,
            2775.488, 2.740, 5.300, 2785.204]
        }, index=range(0, 10))

data_b = pd.DataFrame({
        ''
    })

if __name__ == '__main__':
    print(data_a)
    print(data_a.loc[2,'item'])
    print(data_a.get_value(2,'item'))
    
    print(data_a.lookup([2,4],['item','value']))
---------------------------

# Lesson-5 Pandas Plots
"""
    Pandas' plot () method for Series and DataFrames are a simple wrapper
    around matplotlib's plot() method
    http://pandas.pydata.org/pandas-docs/version/0.15.1/visualization.html
    plot() on a Series
    http://pandas.pydata.org/pandas-docs/dev/generated/pandas.Series.plot.html 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def basic_plotting_series():
    mytimeseries = pd.Series(data=np.random.randn(500),
                             index=pd.date_range('1/1/2000', periods=500))
    print(type(mytimeseries))
    print(mytimeseries)
    mytimeseries = mytimeseries.cumsum() # get cumulative sum over flat array
    mytimeseries.plot()
    plt.show()

def basic_plotting_df():
    """ plot() on a DataFrame; it plots all of the columns with labels
    http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.plot.html
    """
    mytimeseries = pd.Series(data=np.random.randn(500),
                             index=pd.date_range('1/1/2000', periods=500))
    df = pd.DataFrame(data=np.random.randn(500,5),
                      index=mytimeseries.index,
                      columns=list('ABCDE'))
    print(df)
    print(df.head(2))
    df = df.cumsum()
    df.plot(kind='line', use_index=True, style='-')
    plt.show()
    
if __name__ == '__main__':
    basic_plotting_series()
    basic_plotting_df()
-----------------------

# Lesson-6 pandas_timestamp_joins
""" There's two dataframes with dates as their index
    One of the dataframes has a timestamp and this prevents from adding
    the dataframes together.  How can you match up time stamps?"""

from pandas import DataFrame, Timestamp, concat
# Setup dataframe df1

df1 = DataFrame({'col1':[Timestamp('20130102000030'),
                         Timestamp('2013-01-03 00:00:30'),
                         Timestamp('1/4/2013 000030')],
        'col2':[1,10,18]
        })
print("This is dataframe 1\n",df1,"\n")

# Set index for df1
df1 = df1.set_index('col1')
print("Col1 is now the index \n", df1,"\n")    

#Setup dataframe df2
d = {'col2':[22,10,113]}    
i = [Timestamp('20130102'),
     Timestamp('2013-01-03'),
     Timestamp('1/4/2013')]
df2 = DataFrame(data=d, index=i)
df2.index.name = 'col2'
print("This is dataframe 2 \n",df2,"\n")    

#Adding dataframes by rows:
df1 = DataFrame([1,2,3])    
print("New DataFrame 1 \n",df1,"\n")
df2 = DataFrame([4,5,6])
print("New DataFrame 2 \n",df2,"\n")
print("Concatenated df1, df2 \n", concat([df2,df1]),"\n")

# Mergeing data frames by index:
# Setup df1
d = {'col1':[22,10,113]}
i = [Timestamp('1/1/2013'),
     Timestamp('1/2/2013'),
     Timestamp('1/3/2013')]
df1 = DataFrame(data=d, index=i)
print ("Another df1 \n", df1, "\n") # 3 rows (of dates), 1 col of integers

# Setup df2
d = {'col2':[5,5]}
i = [Timestamp('1/1/2013'),
     Timestamp('1/3/2013')]
df2 = DataFrame(data=d, index=i)
print ("Another df2 \n", df2, "\n") # 2 rows (of dates), 1 col of integers

# Merge df1 and df2:
df3 = df1.merge(df2, left_index=True, right_index=True, how='left')
print("Merged data frames is df3 \n",df3,"\n")

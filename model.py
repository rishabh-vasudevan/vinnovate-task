from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
import numpy as np

df = pd.read_excel('Book1.xlsx')
df2 = pd.read_excel('Book2.xlsx')
data2 = []
for i in range(len(df2)):
    data2.append(list(df2.iloc[i]))
data_set=[]
for i in range(len(df)):
    data_set.append(list(df.iloc[i]))
data=[]
for i in data_set:
    for j in range(1,13):
        data.append([int(i[0]),df.columns.values[j],i[j],i[13],0])
for i in data:
    for j in data2:
        if i[0]==j[0] and i[1]==j[1]:
            i[4]=1

x = pd.DataFrame(data = data)
x.columns =['Year','Month','Value','Total','Flood']
label = preprocessing.LabelEncoder()


# code to get the proccessed value of the months
# k = x['Month'].unique()
# k = label.fit_transform(k)
# print(k) 
# month = label.fit_transform(list(x['Month']))

# x['Month']=month
# y = x['Flood']
# x.drop(['Flood','Total'],1,inplace=True)
# x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)
# clf=LogisticRegression()
# clf.fit(x_train,y_train)
# acc = clf.score(x_test,y_test)
# print(acc)
# Year = int(input('Enter the Year: '))
# Month = input('Enter the Month: ')
# Rain = int(input('Enter mm of Rain: '))

# Mon = {'JAN':4,'FEB':3,'MAR':7,'APR':0,'MAY':8,'JUN':6,'JUL':5,'AUG':1,'SEP':11,'OCT':10,'NOV':9,'DEC':2}
# promonth = Mon[Month]
# arr = [Year,promonth,Rain]
# print(clf.predict([[Year,promonth,Rain]]))
# print([[Year,promonth,Rain]])



month = label.fit_transform(list(x['Month']))

x['Month']=month
y = x['Flood']
x.drop(['Flood','Month','Year','Total'],1,inplace=True)
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)
clf=LogisticRegression()
clf.fit(x_train,y_train)
acc = clf.score(x_test,y_test)
print(acc)
Date = int(input('Enter the Date: '))
Month = input('Enter the Month(please enter in caps and only the first three letters of the month)')
Rain = int(input('Enter mm of Rain: '))
Mon = {'JAN':[4,31],'FEB':[3,28],'MAR':[7,31],'APR':[0,30],'MAY':[8,31],'JUN':[6,30],'JUL':[5,31],'AUG':[1,31],'SEP':[11,30],'OCT':[10,31],'NOV':[9,30],'DEC':[2,31]}

#Assuming it will rain constantly for the rest of the month

month_rain = (Rain/Date)*Mon[Month][1]

print(clf.predict([[month_rain]]))

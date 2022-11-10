# -*- coding: utf-8 -*-
"""
Created on Sun May 30 19:01:02 2021

@author: foster-s
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Regression import linreg,stepwise

#Read in the file from specified location
df = pd.read_excel("C:/Users/foster-s/OneDrive - Texas A&M University/Python Projects/Stat 656/HW 1/diamondswmissing.xlsx")

#Create the attribute map, remember that carat is actually capitalized. Other important note is that python actually cares
#about the cut column having the attributes being capitalized. Code didn't work with fair instead had to do Fair. Jesus.
#Decided to ignore observations, but some defined it as an ID, I don't think it really matters. 
attribute_map = { 
    "obs": [DT.Ignore, ("")], 
    "price": [DT.Interval, (300, 20000)], 
    "Carat": [DT.Interval, (0.2, 5.5)], 
    "cut": [DT.Nominal, ("Fair", "Good", "Ideal", "Premium", "Very Good")], 
    "color": [DT.Nominal, ("D", "E", "F", "G", "H", "I", "J")], 
    "clarity": [DT.Nominal, ("I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2")], 
    "depth": [DT.Interval, (40, 80)], 
    "table": [DT.Interval, (40, 100)], 
    "x": [DT.Interval, (0, 11)], 
    "y": [DT.Interval, (0, 60)], 
    "z": [DT.Interval, (0, 32)]}

#This utilizes the ReplaceImputeEncode method seen in the API. You have to label nominal_encoding as one-hot in order
#for it to work.
#Don't include drop because this skews the results.
rie = ReplaceImputeEncode(data_map=attribute_map, display=True, nominal_encoding="one-hot")
encoded_df = rie.fit_transform(df)

#This is the stepwise method and in the API it says reg is defaulted to stepwise, but can choose either forward or backward
#originally tried by denoting reg="stepwise", got an error message saying it had to be linear or logistic so input linear
#The stepwise function requires a "target" variable which some defined before or you can define as the column price as 
#I have done in the below function. 
sw = stepwise(encoded_df, "price", reg="linear")
selected = sw.fit_transform()

#This follows what we did in class except didn't use the drop method as that wasn't shown in the API plus obs is ignored
#in the dictionary already so it isn't being factored into the regression method. 
y = encoded_df["price"]
X = encoded_df[selected]

#running the actual linear regression
lr = LinearRegression()
lr = lr.fit(X,y)

#printing out the linear regression...
linreg.display_coef(lr, X, y, col=X.columns)
linreg.display_metrics(lr, X, y)


import matplotlib
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import statistics
from sklearn import linear_model

featuresdf = pd.read_csv("TrainingSetValues.csv")
labelsdf = pd.read_csv("TrainingSetLabels.csv")

featuresdf['status_group'] = labelsdf['status_group'].values
yvals = []
for i in featuresdf['status_group'].unique():
    s = i.replace(' ', '_')
    yvals.append(s)
    featuresdf[s] = pd.get_dummies(featuresdf['status_group'])[i]

#since status group is cat, change all labels to categorical values

f = featuresdf['status_group'].unique()
def relabel(row):
    for i in range(len(f)):
        if row['status_group'] == f[i]:
            return i
featuresdf['status_group_relabel'] = featuresdf.apply(lambda row: relabel(row), axis = 1)

# do a table for status group and show the mean population
pivot1 = featuresdf.pivot_table(index="status_group",values=["amount_tsh","population", "gps_height"],aggfunc=[np.mean,np.median,statistics.mode])

ax = pivot1.plot.bar(rot = 0)
plt.show()

for i in yvals:
    lreg1 = smf.logit(formula = i + '~ amount_tsh + population', data = featuresdf).fit()
    
    print (lreg1.summary())
    print('')
    
    # odds ratios
    print ("Odds Ratios")
    print (np.exp(lreg1.params))
    
    # odd ratios with 95% confidence intervals
    params = lreg1.params
    conf = lreg1.conf_int()
    conf['OR'] = params
    conf.columns = ['Lower CI', 'Upper CI', 'OR']
    print (np.exp(conf))



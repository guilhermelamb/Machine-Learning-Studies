import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv('credit_data.csv')

#Remove nan
base = base.dropna()

#Outlier -> age
plt.boxplot(base.iloc[:,2], showfliers = True)
outliers_age = base[base['age'] < -20]

#Outlier -> loan
plt.boxplot(base.iloc[:,3])
outliers_loan = base[base['loan'] > 13400]


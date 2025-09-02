# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:38:57 2023

@author: chaorulu
"""

import pandas as pd
import numpy as np

df1=pd.read_csv('Utrecht/Blue_Route_choice_data_3.csv',sep=',')
A=df1.describe()
df2=pd.read_csv('Utrecht/Green_Route_choice_data_3.csv',sep=',')
df3=pd.read_csv('Utrecht/Length_Route_choice_data_3.csv',sep=',')
df4=pd.read_csv('Utrecht/traffic light_Route_choice_data_3.csv',sep=',')
df=pd.read_csv('Utrecht/Route_choice_data_3.csv',sep=',')
df=df.iloc[:, 1:]
df['Selected route']='Selected Route'
df = df.rename(columns={'Total_Average_traffic_light': 'Total_traffic_light'})
df = df.rename(columns={'Total_Average_traffic_sign': 'Total_traffic_sign'})

df['y']=1
df2['y']=0
df3['y']=0
df4['y']=0
df1['y']=0

frames = [df, df1, df2, df3, df4]

result = pd.concat(frames)
result['Average_traffic_light']=result['Total_traffic_light']/result['total_distance']
result['Average_traffic_sign']=result['Total_traffic_sign']/result['total_distance']

result['PS']=1

df1_path=pd.read_csv('Utrecht/Blue_Path_data_3.csv',sep=',')
df2_path=pd.read_csv('Utrecht/Green_Path_data_3.csv',sep=',')
df3_path=pd.read_csv('Utrecht/Length_Path_data_3.csv',sep=',')
df4_path=pd.read_csv('Utrecht/traffic light_Path_data_3.csv',sep=',')
df_path=pd.read_csv('Utrecht/Path_data_3.csv',sep=',')
df_path=df_path.iloc[:, 1:]
df_path['Route Type']='Selected Route'
df_path = df_path.rename(columns={'lines': 'geometry'})

df1_path = df1_path.drop('lines', axis=1)
df2_path = df2_path.drop('lines', axis=1)
df3_path = df3_path.drop('lines', axis=1)
df4_path = df4_path.drop('lines', axis=1)

df1_path = df1_path.drop(index=0).reset_index(drop=True)
df2_path = df2_path.drop(index=0).reset_index(drop=True)
df3_path = df3_path.drop(index=0).reset_index(drop=True)
df4_path = df4_path.drop(index=0).reset_index(drop=True)

frames = [df_path, df1_path, df2_path, df3_path, df4_path]

Path_size = pd.concat(frames)

TripID = df2_path['tripid']
TripID = TripID.drop_duplicates()
TripID = TripID.reset_index(drop=True)

for n in range(len(TripID)):
    #n=0
    testdf=Path_size[Path_size['tripid']==TripID[n]]
    
    #testdf['Route Type'].value_counts()
    #testdf.columns
    
    # Calculate total path length for each path
    path_lengths = testdf.groupby('Route Type')['length'].sum().rename('total_path_length')
    testdf = testdf.merge(path_lengths, on='Route Type')
    
    # Count the number of paths each link appears in
    link_counts = testdf.groupby('geometry')['Route Type'].nunique().rename('link_path_count')
    testdf = testdf.merge(link_counts, on='geometry')
    
    # Calculate Path Size component for each link in each path
    testdf['path_size_component'] = (testdf['length'] / testdf['total_path_length']) * (1 / testdf['link_path_count'])
    
    # Sum the components to get Path Size for each path
    path_size = testdf.groupby('Route Type')['path_size_component'].sum().rename('path_size')
    
    result.loc[(result['Selected route'] == 'Blue') & (result['TripID'] == TripID[n]), 'PS']=path_size['Blue']
    result.loc[(result['Selected route'] == 'Green') & (result['TripID'] == TripID[n]), 'PS']=path_size['Green']
    result.loc[(result['Selected route'] == 'Length') & (result['TripID'] == TripID[n]), 'PS']=path_size['Length']
    result.loc[(result['Selected route'] == 'traffic light') & (result['TripID'] == TripID[n]), 'PS']=path_size['traffic light']
    result.loc[(result['Selected route'] == 'Selected Route') & (result['TripID'] == TripID[n]), 'PS']=path_size['Selected Route']
    print (n)
    


# A=df4.describe()
# A.to_csv('Utrecht/descriptive statistics_data.csv')

result.to_csv('Utrecht/Merged_Route_choice_data_3.csv')

result['y'].value_counts()

result['Selected route'].value_counts()



#Path Size Logit

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
result=pd.read_csv('Utrecht/Merged_Route_choice_data_3.csv',sep=',')

result = result.iloc[: , 1:]

result['ln(PS)']=np.log(result['PS'])

y=result['y']



#Variable selection
Variable=['total_distance','Max_road','Average_road', 'Min_road','Max_sidewalk','Average_sidewalk','Min_sidewalk', 'Average_building','Average_wall', 'Average_fence', 'Total_Pole_Coverage', 
    'Average_traffic_light','Max_vegetation',  'Average_vegetation','Min_vegetation', 'Max_sky','Average_sky', 'Min_sky','Average_person',
    'Average_car', 'Average_truck', 'Average_bus','Average_train', 'Average_motorcycle', 'Average_bicycle', 'ln(PS)']

Model=result[Variable]

Model = Model.fillna(Model.mean())

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = Model.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(Model.values, i)
                          for i in range(len(Model.columns))]

print(vif_data)


#Selected Variable
#model 1
Variable=['total_distance','Average_road', 'Min_road','Average_sidewalk','Min_sidewalk', 'Average_building','Average_wall', 'Average_fence', 'Total_Pole_Coverage', 
    'Average_traffic_light','Max_vegetation',  'Average_vegetation','Min_vegetation', 'Min_sky','Average_person',
    'Average_car', 'Average_truck', 'Average_bus','Average_train', 'Average_motorcycle', 'Average_bicycle', 'ln(PS)']

Model=result[Variable]

Model = Model.fillna(Model.mean())

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = Model.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(Model.values, i)
                          for i in range(len(Model.columns))]

print(vif_data)

#Model2
Variable=['total_distance','Average_road', 'Min_road','Average_sidewalk','Min_sidewalk', 'Average_building','Average_wall', 'Total_Pole_Coverage', 
    'Average_traffic_light','Max_vegetation',  'Average_vegetation', 'Min_sky','Average_person',
    'Average_car', 'Average_train', 'Average_motorcycle', 'Average_bicycle', 'ln(PS)']


Model=result[Variable]

Model = Model.fillna(Model.mean())

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = Model.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(Model.values, i)
                          for i in range(len(Model.columns))]

print(vif_data)

#Model 1
Variable=['total_distance','Average_road', 'Min_road','Average_sidewalk','Min_sidewalk', 'Average_building','Average_wall', 'Average_fence', 'Total_Pole_Coverage', 
    'Average_traffic_light','Max_vegetation',  'Average_vegetation','Min_vegetation',  'Min_sky','Average_person',
    'Average_car', 'Average_truck', 'Average_bus','Average_train', 'Average_motorcycle', 'Average_bicycle', 'ln(PS)']


Model=result[Variable]

Model = Model.fillna(Model.mean())

#Dis=Model['total_distance']/1000
PS=Model['ln(PS)']

#Model 1
# # Standardize features
scaler = MinMaxScaler()
Model[:-1] = scaler.fit_transform(Model[:-1])
#Model['total_distance']=Dis
Model['ln(PS)']=PS

# Add a constant for the intercept in statsmodels
X = Model#sm.add_constant(Model)

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
Logit_result = logit_model.fit()
AIC=Logit_result.aic


# Print the summary of the model
print(Logit_result.summary())
print(AIC)

#Model 2
Variable=['total_distance','Average_road', 'Min_road','Average_sidewalk','Min_sidewalk', 'Average_building','Average_wall', 'Total_Pole_Coverage', 
    'Average_traffic_light','Max_vegetation',  'Average_vegetation', 'Min_sky','Average_person',
    'Average_car', 'Average_train', 'Average_motorcycle', 'Average_bicycle', 'ln(PS)']

Model=result[Variable]

Model = Model.fillna(Model.mean())

# # Standardize features
scaler = MinMaxScaler()
Model[:-1] = scaler.fit_transform(Model[:-1])
#Model['total_distance']=Dis
Model['ln(PS)']=PS

#Model = scaler.fit_transform(Model)
# Add a constant for the intercept in statsmodels
X = Model#sm.add_constant(Model)

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
Logit_result = logit_model.fit()
AIC=Logit_result.aic
# Print the summary of the model
print(Logit_result.summary())
print(AIC)

#Model 3
Variable=['total_distance','Average_road', 'Min_road','Average_sidewalk','Min_sidewalk', 'Average_building','Average_wall', 'Total_Pole_Coverage', 
    'Average_traffic_light','Max_vegetation',  'Average_vegetation', 'Min_sky','Average_person',
    'Average_car', 'Average_train', 'Average_motorcycle', 'Average_bicycle']

Model=result[Variable]

Model = Model.fillna(Model.mean())

# # Standardize features
scaler = MinMaxScaler()
Model[:-1] = scaler.fit_transform(Model[:-1])
#Model['total_distance']=Dis

#Model = scaler.fit_transform(Model)
# Add a constant for the intercept in statsmodels
X = Model#sm.add_constant(Model)

# Fit the logistic regression model
logit_model = sm.Logit(y, X)
Logit_result = logit_model.fit()
AIC=Logit_result.aic
# Print the summary of the model
print(Logit_result.summary())
print(AIC)


































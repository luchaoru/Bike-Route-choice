# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:19:46 2024

@author: chaorulu
"""

#Read gdf_edges_no_dup

import geopandas as gpd
import pandas as pd
import osmnx as ox
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
import matplotlib.pyplot as plt
import requests
import os
import numpy as np
import math


# Read data
gdf_edges_no_dup=gpd.read_file("C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/city with built Env.geojson")

gdf_edges_no_dup['lines']=gdf_edges_no_dup['geometry']

#data
df=pd.read_csv('UtrechtZwolle.csv')
df['date']=df['timestamp']
df['timestamp']=pd.to_datetime(df['timestamp']+''+df['Hour'].astype(str)+''+df['Minute'].astype(str)+''
                               +df['Second'].astype(str), format='%d %b %y%H%M%S')

df.columns
df.head()

newdf = df.drop_duplicates( subset = ['tripid', 'Hour', 'Minute','date'], keep = 'first').reset_index(drop = True)
newdf = newdf[newdf['longitude']<5.702920]

geoBike = gpd.GeoDataFrame(newdf, geometry=gpd.points_from_xy(newdf.longitude, newdf.latitude),crs="EPSG:4326")


# Perform spatial join: points to linestrings, based on proximity (nearest point to each line)
#geoBike_merged = gpd.sjoin(geoBike, gdf_edges_no_dup, how="left", op='within')

# Remove any pre-existing 'index_left' and 'index_right' columns from geoBike and gdf_edges_no_dup
geoBike = geoBike.drop(columns=['index_left', 'index_right'], errors='ignore')
gdf_edges_no_dup = gdf_edges_no_dup.drop(columns=['index_left', 'index_right', 'distance'], errors='ignore')

# Align CRS if necessary
if geoBike.crs != gdf_edges_no_dup.crs:
    gdf_edges_no_dup = gdf_edges_no_dup.to_crs(geoBike.crs)

geoBike_merged = gpd.sjoin_nearest(geoBike, gdf_edges_no_dup, how='left', distance_col='distance')

COLUMN_NAMES=['TripID','Selected route','Origin','Destination','Start_time','End_Time','total_distance','total_travel_time', 'Max_road','Average_road',
    'Min_road','Max_sidewalk','Average_sidewalk','Min_sidewalk','Max_building','Average_building','Min_building',
    'Max_wall','Average_wall', 'Min_wall','Max_fence', 'Average_fence', 'Min_fence', 'Pole_Coverage', 'Max_traffic_light',
    'Average_traffic_light','Min_traffic_light','Max_traffic_sign','Average_traffic_sign', 'Min_traffic_sign', 'Max_vegetation',  'Average_vegetation',
    'Min_vegetation', 'Max_terrain','Average_terrain','Min_terrain','Max_sky', 'Average_sky', 'Min_sky', 'Max_person', 'Average_person','Min_person',
    'Max_rider', 'Average_rider', 'Min_rider', 'Max_car', 'Average_car', 'Min_car', 'Max_truck', 'Average_truck', 'Min_truck', 'Max_bus',
    'Average_bus','Min_bus','Max_train','Average_train', 'Min_train','Max_motorcycle','Average_motorcycle','Min_motorcycle', 'Max_bicycle',
    'Average_bicycle', 'Min_bicycle','Route GeoInfo']


Route_choice_data = pd.DataFrame([[np.nan] * len(COLUMN_NAMES)], columns=COLUMN_NAMES)
df=Route_choice_data

Path_data = pd.DataFrame([[np.nan] * len(['tripid','length','lines'])], columns=['tripid','length','lines'])

TripID=geoBike['tripid']
TripID = TripID.drop_duplicates()
TripID = TripID.reset_index(drop=True)

for n in range(len(TripID)):
    #n=1
    testdf=geoBike_merged[geoBike_merged['tripid']==TripID[n]]
    testdf=testdf.sort_values(by=['timestamp'])
    testdf = testdf.reset_index(drop=True) 
    
    df.loc[n,'TripID']=TripID[n]
    df.loc[n,'Selected route']=1
    df.loc[n,'Origin']=testdf['geometry'][0]
    df.loc[n,'Destination']=testdf['geometry'].iloc[-1]
    df.loc[n,'Start_time']=testdf['timestamp'].iloc[0]
    df.loc[n,'End_Time']=testdf['timestamp'].iloc[-1]
    df.loc[n,'total_travel_time']=(testdf['timestamp'].iloc[-1]-testdf['timestamp'].iloc[0]).total_seconds()#second
    
    testdf_no_dup = testdf.drop_duplicates(subset='lines')
    testdf_no_dup = testdf_no_dup.reset_index(drop=True)     
    
    Path = testdf_no_dup[['tripid','length','lines']]
    
    df.loc[n,'total_distance']=sum(testdf_no_dup['length'])#meter
    
    df.loc[n,'Average_road']=testdf['road'].mean()
    df.loc[n,'Max_road']=max(testdf['road'])
    df.loc[n,'Min_road']=min(testdf['road'])
    
    df.loc[n,'Average_sidewalk']=testdf['sidewalk'].mean()
    df.loc[n,'Max_sidewalk']=max(testdf['sidewalk'])
    df.loc[n,'Min_sidewalk']=min(testdf['sidewalk'])
    
    df.loc[n,'Average_building']=testdf['building'].mean()
    df.loc[n,'Max_building']=max(testdf['building'])
    df.loc[n,'Min_building']=min(testdf['building'])
    
    df.loc[n,'Average_wall']=testdf['wall'].mean()
    df.loc[n,'Max_wall']=max(testdf['wall'])
    df.loc[n,'Min_wall']=min(testdf['wall'])
    
    df.loc[n,'Average_fence']=testdf['fence'].mean()
    df.loc[n,'Max_fence']=max(testdf['fence'])
    df.loc[n,'Min_fence']=min(testdf['fence'])
    
    df.loc[n,'Total_Pole_Coverage']=testdf['pole'].mean() #pole coverage
    
    df.loc[n,'Total_traffic_light']=testdf['traffic light'].mean()
    df.loc[n,'Max_traffic_light']=max(testdf['traffic light'])
    df.loc[n,'Min_traffic_light']=min(testdf['traffic light'])
    
    df.loc[n,'Total_traffic_sign']=testdf['traffic sign'].sum()
    df.loc[n,'Max_traffic_sign']=max(testdf['traffic sign'])
    df.loc[n,'Min_traffic_sign']=min(testdf['traffic sign'])
    
    df.loc[n,'Average_vegetation']=testdf['vegetation'].mean()
    df.loc[n,'Max_vegetation']=max(testdf['vegetation'])
    df.loc[n,'Min_vegetation']=min(testdf['vegetation'])
    
    df.loc[n,'Average_terrain']=testdf['terrain'].mean()
    df.loc[n,'Max_terrain']=max(testdf['terrain'])
    df.loc[n,'Min_terrain']=min(testdf['terrain'])
    
    df.loc[n,'Average_sky']=testdf['sky'].mean()
    df.loc[n,'Max_sky']=max(testdf['sky'])
    df.loc[n,'Min_sky']=min(testdf['sky'])
    
    df.loc[n,'Average_person']=testdf['person'].mean()
    df.loc[n,'Max_person']=max(testdf['person'])
    df.loc[n,'Min_person']=min(testdf['person'])
    
    df.loc[n,'Average_rider']=testdf['rider'].mean()
    df.loc[n,'Max_rider']=max(testdf['rider'])
    df.loc[n,'Min_rider']=min(testdf['rider'])
    
    df.loc[n,'Average_car']=testdf['car'].mean()
    df.loc[n,'Max_car']=max(testdf['car'])
    df.loc[n,'Min_car']=min(testdf['car'])
    
    df.loc[n,'Average_truck']=testdf['truck'].mean()
    df.loc[n,'Max_truck']=max(testdf['truck'])
    df.loc[n,'Min_truck']=min(testdf['truck'])
    
    df.loc[n,'Average_bus']=testdf['bus'].mean()
    df.loc[n,'Max_bus']=max(testdf['bus'])
    df.loc[n,'Min_bus']=min(testdf['bus'])
    
    df.loc[n,'Average_train']=testdf['train'].mean()
    df.loc[n,'Max_train']=max(testdf['train'])
    df.loc[n,'Min_train']=min(testdf['train'])
    
    df.loc[n,'Average_motorcycle']=testdf['motorcycle'].mean()
    df.loc[n,'Max_motorcycle']=max(testdf['motorcycle'])
    df.loc[n,'Min_motorcycle']=min(testdf['motorcycle'])
    
    df.loc[n,'Average_bicycle']=testdf['bicycle'].mean()
    df.loc[n,'Max_bicycle']=max(testdf['bicycle'])
    df.loc[n,'Min_bicycle']=min(testdf['bicycle'])
    
    #Route_choice_data = pd.concat([Route_choice_data, df], ignore_index=True)
    Path_data = pd.concat([Path_data, Path], ignore_index=True)
    print (n)

Route_choice_data=df
Route_choice_data.to_csv("C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/Route_choice_data_3.csv") 
Path_data.to_csv("C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/Path_data_3.csv") 


from shapely.wkt import loads 

#find other routes
Route_choice_data = pd.read_csv("C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/Route_choice_data.csv")
CITYG =ox.load_graphml('C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/city with built Env.graphml')
backUp=Route_choice_data

Route_choice_data['Origin'] = Route_choice_data['Origin'].apply(loads)
Route_choice_data['Destination'] = Route_choice_data['Destination'].apply(loads)

Other_Route_choice_data = pd.DataFrame([[np.nan] * len(COLUMN_NAMES)], columns=COLUMN_NAMES)
df=Other_Route_choice_data
Path_data = pd.DataFrame([[np.nan] * len(['tripid','length','lines'])], columns=['tripid','length','lines'])

Route_type='traffic light' # types 'Blue','Green','Length','traffic light'

for n in range(len(Route_choice_data)):
    #n=1
    O=Route_choice_data['Origin'][n]
    D=Route_choice_data['Destination'][n]
    
    #Find the nearest point on the map
    nearest_O = ox.distance.nearest_nodes(CITYG, X=O.x, Y=O.y)
    nearest_D = ox.distance.nearest_nodes(CITYG, X=D.x, Y=D.y)
    
    if nx.has_path(CITYG, nearest_O, nearest_D):
    
        shortest_path = nx.dijkstra_path(CITYG, source=nearest_O, target=nearest_D, weight=Route_type)
     
        # Extract the attributes of the edges along the shortest path
        path_attributes = []
        for i in range(len(shortest_path) - 1):
            u, v = shortest_path[i], shortest_path[i + 1]
            edge_data = CITYG[u][v]  # Get the edge data between nodes u and v
            # Add relevant edge attributes to the list
            path_attributes.append({
                **edge_data  # This includes all attributes of the edge
            })
    
        # Create a DataFrame from the list of attributes
        testdf=[]
        for i in range(len(shortest_path) - 1):
            testdf.append(path_attributes[i][0])
    
        testdf = pd.DataFrame(testdf)
        
        if len(testdf)>0:
          
            Path = testdf[['length','geometry']]
            Path['tripid']=Route_choice_data['TripID'][n]
            
            df.loc[n,'TripID']=Route_choice_data['TripID'][n]
            df.loc[n,'Selected route']=Route_type
            df.loc[n,'Origin']=O
            df.loc[n,'Destination']=D
            df.loc[n,'Start_time']=Route_choice_data['Start_time'][n]
            df.loc[n,'End_Time']=Route_choice_data['End_Time'][n]
            df.loc[n,'total_travel_time']=np.nan#second
            
            df.loc[n,'total_distance']=sum(testdf['length'])#meter
            
            df.loc[n,'Average_road']=testdf['road'].mean()
            df.loc[n,'Max_road']=max(testdf['road'])
            df.loc[n,'Min_road']=min(testdf['road'])
            
            df.loc[n,'Average_sidewalk']=testdf['sidewalk'].mean()
            df.loc[n,'Max_sidewalk']=max(testdf['sidewalk'])
            df.loc[n,'Min_sidewalk']=min(testdf['sidewalk'])
            
            df.loc[n,'Average_building']=testdf['building'].mean()
            df.loc[n,'Max_building']=max(testdf['building'])
            df.loc[n,'Min_building']=min(testdf['building'])
            
            df.loc[n,'Average_wall']=testdf['wall'].mean()
            df.loc[n,'Max_wall']=max(testdf['wall'])
            df.loc[n,'Min_wall']=min(testdf['wall'])
            
            df.loc[n,'Average_fence']=testdf['fence'].mean()
            df.loc[n,'Max_fence']=max(testdf['fence'])
            df.loc[n,'Min_fence']=min(testdf['fence'])
            
            df.loc[n,'Total_Pole_Coverage']=testdf['pole'].mean() #pole coverage
            
            df.loc[n,'Total_traffic_light']=testdf['traffic light'].mean()
            df.loc[n,'Max_traffic_light']=max(testdf['traffic light'])
            df.loc[n,'Min_traffic_light']=min(testdf['traffic light'])
            
            df.loc[n,'Total_traffic_sign']=testdf['traffic sign'].mean()
            df.loc[n,'Max_traffic_sign']=max(testdf['traffic sign'])
            df.loc[n,'Min_traffic_sign']=min(testdf['traffic sign'])
            
            df.loc[n,'Average_vegetation']=testdf['vegetation'].mean()
            df.loc[n,'Max_vegetation']=max(testdf['vegetation'])
            df.loc[n,'Min_vegetation']=min(testdf['vegetation'])
            
            df.loc[n,'Average_terrain']=testdf['terrain'].mean()
            df.loc[n,'Max_terrain']=max(testdf['terrain'])
            df.loc[n,'Min_terrain']=min(testdf['terrain'])
            
            df.loc[n,'Average_sky']=testdf['sky'].mean()
            df.loc[n,'Max_sky']=max(testdf['sky'])
            df.loc[n,'Min_sky']=min(testdf['sky'])
            
            df.loc[n,'Average_person']=testdf['person'].mean()
            df.loc[n,'Max_person']=max(testdf['person'])
            df.loc[n,'Min_person']=min(testdf['person'])
            
            df.loc[n,'Average_rider']=testdf['rider'].mean()
            df.loc[n,'Max_rider']=max(testdf['rider'])
            df.loc[n,'Min_rider']=min(testdf['rider'])
            
            df.loc[n,'Average_car']=testdf['car'].mean()
            df.loc[n,'Max_car']=max(testdf['car'])
            df.loc[n,'Min_car']=min(testdf['car'])
            
            df.loc[n,'Average_truck']=testdf['truck'].mean()
            df.loc[n,'Max_truck']=max(testdf['truck'])
            df.loc[n,'Min_truck']=min(testdf['truck'])
            
            df.loc[n,'Average_bus']=testdf['bus'].mean()
            df.loc[n,'Max_bus']=max(testdf['bus'])
            df.loc[n,'Min_bus']=min(testdf['bus'])
            
            df.loc[n,'Average_train']=testdf['train'].mean()
            df.loc[n,'Max_train']=max(testdf['train'])
            df.loc[n,'Min_train']=min(testdf['train'])
            
            df.loc[n,'Average_motorcycle']=testdf['motorcycle'].mean()
            df.loc[n,'Max_motorcycle']=max(testdf['motorcycle'])
            df.loc[n,'Min_motorcycle']=min(testdf['motorcycle'])
            
            df.loc[n,'Average_bicycle']=testdf['bicycle'].mean()
            df.loc[n,'Max_bicycle']=max(testdf['bicycle'])
            df.loc[n,'Min_bicycle']=min(testdf['bicycle'])
            
            Path_data = pd.concat([Path_data, Path], ignore_index=True)
        #Other_Route_choice_data = pd.concat([Other_Route_choice_data, df], ignore_index=True)
        print(n)
    
Path_data['Route Type']=Route_type
Other_Route_choice_data=df

# Define the directory and filename
directory = "C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht"
filename = Route_type+'_Route_choice_data_3.csv'
filename2 = Route_type+'_Path_data_3.csv'

# Combine to create the full path
file_path = os.path.join(directory, filename)
Other_Route_choice_data.to_csv(file_path, index=False)

file_path2 = os.path.join(directory, filename2)
Path_data.to_csv(file_path2, index=False)

# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:13:59 2024

@author: chaorulu
"""

import osmnx as ox

# Download by a bounding box
# bounds = (17.4110711999999985,18.4494298999999984,59.1412578999999994,59.8280297000000019)
# x1,x2,y1,y2 = bounds
# boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
# G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
# start_time = time.time()
# save_graph_shapefile_directional(G, filepath='./network-new')
# print("--- %s seconds ---" % (time.time() - start_time))

# Download by place name
places = ['Utrecht, Netherlands']
G = ox.graph_from_place(places,network_type='bike', simplify=True) 
G = ox.distance.add_edge_lengths(G, precision=3, edges=None)
fig, ax = ox.plot_graph(G, edge_linewidth=0.8)
##add elevation
#G = ox.elevation.add_node_elevations_google(G, api_key='AIzaSyDmKEbiHO7y0oMvVcfCxIcsbiohK3xx0-8')
#G = ox.elevation.add_edge_grades(G)
# Save geo file
ox.io.save_graphml(G, filepath='C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/city.graphml', gephi=False, encoding='utf-8')


import geopandas as gpd
import pandas as pd
import osmnx as ox
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
from pandana.loaders import osm
import matplotlib.pyplot as plt
import requests
import os
import numpy as np
import math

#Read geo file
G = ox.load_graphml('C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/city.graphml')

####Converting MultiDiGraph to GeoDataFrame
# convert multidigraph edges to geodataframe
gdf_edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True,node_geometry=False, fill_edge_geometry=True)

# convert multidigraph nodes to geodataframe
gdf_nodes = ox.utils_graph.graph_to_gdfs(G, nodes=True, edges=False,node_geometry=True, fill_edge_geometry=False)

# apply projection
gdf_nodes['geometry'] = gdf_nodes.geometry.to_crs("EPSG:4326")
# # update x and y coordinates with projected coordinates
gdf_nodes['x'] = gdf_nodes['geometry'].x
gdf_nodes['y'] = gdf_nodes['geometry'].y
# convert edges to projected coordinates
gdf_edges['geometry'] = gdf_edges.geometry.to_crs("EPSG:4326")

#integrate built environment info. to road network
BuiltEnv=pd.read_csv('C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/UtrechtZwolle/Processed_Data_clean_standarized.csv')
BuiltEnv=BuiltEnv.iloc[:,1:]
BuiltEnv = BuiltEnv.fillna(0)
geoBuiltEnv = gpd.GeoDataFrame(BuiltEnv, geometry=gpd.points_from_xy(BuiltEnv.lon, BuiltEnv.lat),crs="EPSG:4326")

# Ensure the CRS (Coordinate Reference System) is the same for both GeoDataFrames
geoBuiltEnv = geoBuiltEnv.to_crs(gdf_edges.crs) 

# Perform spatial join: points to linestrings, based on proximity (nearest point to each line)
gdf_edges_merged = gpd.sjoin_nearest(gdf_edges, geoBuiltEnv, how="left", distance_col="distance")

gdf_edges_no_dup = gdf_edges_merged.drop_duplicates(subset='geometry')

Target_varibles=['road', 'sidewalk','building', 'wall','fence', 
       'pole','traffic light','traffic sign','vegetation', 'terrain', 'sky',
       'person', 'rider', 'car', 'truck', 'bus','train','motorcycle','bicycle']

gdf_edges_no_dup[Target_varibles]=0


import warnings
warnings.filterwarnings("ignore")

for n in range(len(gdf_edges_no_dup)):
# Define the geometry you want to slice by
    #n=0
    target_geometry = gdf_edges_no_dup['geometry'].iloc[n]
    
    # Slice GeoDataFrame to include only rows with that geometry
    gdf_matching = gdf_edges_merged[gdf_edges_merged.geometry == target_geometry]
    DIS=gdf_matching['length'].mean()
    #average rate/distance
    gdf_edges_no_dup['road'].iloc[n]=gdf_matching['road'].mean()/DIS
    gdf_edges_no_dup['sidewalk'].iloc[n]=gdf_matching['sidewalk'].mean()/DIS
    gdf_edges_no_dup['building'].iloc[n]=gdf_matching['building'].mean()/DIS
    gdf_edges_no_dup['wall'].iloc[n]=gdf_matching['wall'].mean()/DIS
    
    gdf_edges_no_dup['fence'].iloc[n]=gdf_matching['fence'].mean()/DIS
    gdf_edges_no_dup['pole'].iloc[n]=sum(gdf_matching['pole'].apply(math.ceil))/len(gdf_matching['pole'])
    gdf_edges_no_dup['traffic light'].iloc[n]=sum(gdf_matching['traffic light'].apply(math.ceil))/len(gdf_matching['traffic light'])
    gdf_edges_no_dup['traffic sign'].iloc[n]=sum(gdf_matching['traffic sign'].apply(math.ceil))/len(gdf_matching['traffic sign'])
    
    gdf_edges_no_dup['vegetation'].iloc[n]=gdf_matching['vegetation'].mean()/DIS
    gdf_edges_no_dup['terrain'].iloc[n]=gdf_matching['terrain'].mean()/DIS
    gdf_edges_no_dup['sky'].iloc[n]=gdf_matching['sky'].mean()/DIS
    gdf_edges_no_dup['person'].iloc[n]=gdf_matching['person'].mean()/DIS
    
    gdf_edges_no_dup['rider'].iloc[n]=gdf_matching['rider'].mean()/DIS
    gdf_edges_no_dup['car'].iloc[n]=gdf_matching['car'].mean()/DIS
    gdf_edges_no_dup['truck'].iloc[n]=gdf_matching['truck'].mean()/DIS
    gdf_edges_no_dup['bus'].iloc[n]=gdf_matching['bus'].mean()/DIS
    
    gdf_edges_no_dup['train'].iloc[n]=gdf_matching['train'].mean()/DIS
    gdf_edges_no_dup['motorcycle'].iloc[n]=gdf_matching['motorcycle'].mean()/DIS
    gdf_edges_no_dup['bicycle'].iloc[n]=gdf_matching['bicycle'].mean()/DIS
    
    print(n)

backUp=gdf_edges_no_dup

for col in gdf_edges_no_dup.columns:
    gdf_edges_no_dup[col] = gdf_edges_no_dup[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)

gdf_edges_no_dup['Green']=(1- gdf_edges_no_dup['vegetation']).clip(lower=0)
gdf_edges_no_dup['Blue']=(1-gdf_edges_no_dup['sky']).clip(lower=0)


# for col in gdf_edges_no_dup.columns:
#     if gdf_edges_no_dup[col].dtype == 'object':
#         print(f"Checking column: {col}")
#         print(gdf_edges_no_dup[col].apply(type).value_counts())  # Print the data types in the column
        
# print(gdf_edges_no_dup.dtypes)    

gdf_edges_no_dup.to_file("C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/city with built Env.geojson", driver="GeoJSON") 
###Convert GeoDataFrames to MultiDiGraph
# set crs attribute
graph_attrs = {"crs": "EPSG:4326"}

# convert geodataframe to multidigraph
proj_multidigraph = ox.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges_no_dup, graph_attrs=graph_attrs)

G=proj_multidigraph
# Fix attributes with comma-separated values
for u, v, key, data in G.edges(keys=True, data=True):
    for attr, value in data.items():
        if isinstance(value, str) and "," in value:  # Check for comma-separated values
            values = value.split(",")  # Split into a list

            # Try to convert to integer
            try:
                data[attr] = int(values[0])  # Keep only the first integer
                continue  # Skip further checks if conversion is successful
            except ValueError:
                pass  # If conversion to int fails, try boolean

            # Try to convert to boolean
            try:
                data[attr] = bool(values[0].strip() == "True")  # Keep only the first boolean value
            except ValueError:
                data[attr] = value  # Keep as a string if all conversions fail

ox.io.save_graphml(G, filepath='C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/Utrecht/city with built Env2.graphml')

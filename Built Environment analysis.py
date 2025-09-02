# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 20:24:11 2023

@author: chaorulu
"""
import os
import torch
from PIL import Image
from IPython.display import display
import pandas as pd
import numpy as np
from fastseg import MobileV3Large
from fastseg.image import colorize, blend

Label_color = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    [  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
    [  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
    [  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
    [  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
    [  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ],
    [  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ],
    [  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ],
    [  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ],
    [  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ],
    [  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ],
    [  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ],
    [  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ],
    [  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ],
    [  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ],
    [  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ],
    [  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ],
    [  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ],
    [  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ],
    [  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ],
    [  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ],
    [  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ],
    [  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ],
    [  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ],
    [  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ],
    [  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ],
    [  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ],
    [  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ],
    [  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ],
    [  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ],
    [  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ],
    [  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ],
    [  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ],
    [  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ],
    [  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ],
    [  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ]]
Label= pd.DataFrame(Label_color, columns =[ 'name', 'id', 'trainId', 'category', 'catId', 'hasInstances', 'ignoreInEval', 'color'])

print(torch.__version__)

model = MobileV3Large.from_pretrained().cpu().eval()

Coordinates=pd.read_csv('Coordinates.csv',sep=',',dtype='str')
Coordinates=Coordinates.iloc[:,1:4]

Folder_directory='C:/Users/chaorulu/OneDrive - OsloMet/Skrivebord/Bike/Code/Bike routing/UtrechtZwolle'

col_name=['lat', 'lon','unlabeled' ,'road', 'sidewalk','building', 'wall','fence', 
       'pole','traffic light','traffic sign','vegetation', 'terrain', 'sky',
       'person', 'rider', 'car', 'truck', 'bus','train','motorcycle','bicycle','license plate']
Processed_Data= pd.DataFrame(columns=col_name,index=range(Coordinates.shape[0]))

places='Utrecht'
#n=4800
for n in range(Coordinates.shape[0]):  
 
    X=Coordinates['lat'][n]
    Y=Coordinates['lon'][n]
    Processed_Data['lat'][n]=X
    Processed_Data['lon'][n]=Y
    Location= str(X)+','+str(Y)
    #Location= '52.045064347827754,5.175217218227303'
    name= Location #Name of the Pic file
    B= Label[['trainId','name']]
    B['name'].loc[B['trainId']==255]='Unlabeled'
    B=B.drop_duplicates(subset=['trainId'])
    B=B.reset_index()
    Avg_share=B[['trainId','name']]
    Avg_share['average_share']=np.nan
    Avg_share['Share_0']=np.nan
    Avg_share['Share_90']=np.nan
    Avg_share['Share_180']=np.nan
    Avg_share['Share_270']=np.nan
    B=B[['trainId','name']]
    #angle 0
    angle=0
    base_filename='Pic_'+name+'_Angle_'+str(angle)
    suffix = '.jpg'
    pic_path = os.path.join(Folder_directory, base_filename + suffix)
    
    if os.path.isfile(pic_path):
    
        img = Image.open(pic_path)
       # display(img.resize((800, 400)))
        
        labels = model.predict_one(img)
    # print('Shape:', labels.shape)
    # print(labels)
    
    # colorized = colorize(labels)
    # display(colorized.resize((800, 400)))     
    
    # composited = blend(img, colorized)
    # display(composited.resize((800, 400)))
    
        unique, counts= np.unique(labels, return_counts=True)
        data_tuples = list(zip(unique, counts))
        list(Label.columns)
        A= pd.DataFrame(data_tuples, columns=['trainId','Frequency'])
        A['Share_0']=A['Frequency']/sum(A['Frequency'])
        C1= B.merge(A, how='left', on='trainId')
        C1=C1[['trainId','name','Share_0']]
        Avg_share.loc[Avg_share['name']=='Unlabeled','Share_0']=C1.loc[Avg_share['name']=='Unlabeled','Share_0']
        Avg_share.loc[Avg_share['name']=='person','Share_0']=C1.loc[Avg_share['name']=='person','Share_0']
        Avg_share.loc[Avg_share['name']=='road','Share_0']=C1.loc[Avg_share['name']=='road','Share_0']
        Avg_share.loc[Avg_share['name']=='sidewalk','Share_0']=C1.loc[Avg_share['name']=='sidewalk','Share_0']
        Avg_share.loc[Avg_share['name']=='building','Share_0']=C1.loc[Avg_share['name']=='building','Share_0']
        Avg_share.loc[Avg_share['name']=='wall','Share_0']=C1.loc[Avg_share['name']=='wall','Share_0']
        Avg_share.loc[Avg_share['name']=='fence','Share_0']=C1.loc[Avg_share['name']=='fence','Share_0']
        Avg_share.loc[Avg_share['name']=='pole','Share_0']=C1.loc[Avg_share['name']=='pole','Share_0']
        Avg_share.loc[Avg_share['name']=='traffic light','Share_0']=C1.loc[Avg_share['name']=='traffic light','Share_0']
        Avg_share.loc[Avg_share['name']=='traffic sign','Share_0']=C1.loc[Avg_share['name']=='traffic sign','Share_0']
        Avg_share.loc[Avg_share['name']=='vegetation','Share_0']=C1.loc[Avg_share['name']=='vegetation','Share_0']
        Avg_share.loc[Avg_share['name']=='terrain','Share_0']=C1.loc[Avg_share['name']=='terrain','Share_0']
        Avg_share.loc[Avg_share['name']=='sky','Share_0']=C1.loc[Avg_share['name']=='sky','Share_0']
        Avg_share.loc[Avg_share['name']=='rider','Share_0']=C1.loc[Avg_share['name']=='rider','Share_0']
        Avg_share.loc[Avg_share['name']=='car','Share_0']=C1.loc[Avg_share['name']=='car','Share_0']
        Avg_share.loc[Avg_share['name']=='truck','Share_0']=C1.loc[Avg_share['name']=='truck','Share_0']
        Avg_share.loc[Avg_share['name']=='bus','Share_0']=C1.loc[Avg_share['name']=='bus','Share_0']
        Avg_share.loc[Avg_share['name']=='train','Share_0']=C1.loc[Avg_share['name']=='train','Share_0']
        Avg_share.loc[Avg_share['name']=='motorcycle','Share_0']=C1.loc[Avg_share['name']=='motorcycle','Share_0']
        Avg_share.loc[Avg_share['name']=='bicycle','Share_0']=C1.loc[Avg_share['name']=='bicycle','Share_0']
        Avg_share.loc[Avg_share['name']=='license plate','Share_0']=C1.loc[Avg_share['name']=='license plate','Share_0']
     
    #angle 90
    angle=90
    base_filename='Pic_'+name+'_Angle_'+str(angle)
    suffix = '.jpg'
    pic_path = os.path.join(Folder_directory, base_filename + suffix)
    
    if os.path.isfile(pic_path):
    
        img = Image.open(pic_path)
        #display(img.resize((800, 400)))
        labels = model.predict_one(img)
        unique, counts= np.unique(labels, return_counts=True)
        data_tuples = list(zip(unique, counts))
        list(Label.columns)
        A= pd.DataFrame(data_tuples, columns=['trainId','Frequency'])
        A['Share_90']=A['Frequency']/sum(A['Frequency'])
        C2= B.merge(A, how='left', on='trainId')
        C2= C2[['trainId','name','Share_90']]
        
        Avg_share.loc[Avg_share['name']=='Unlabeled','Share_90']=C2.loc[Avg_share['name']=='Unlabeled','Share_90']
        Avg_share.loc[Avg_share['name']=='person','Share_90']=C2.loc[Avg_share['name']=='person','Share_90']
        Avg_share.loc[Avg_share['name']=='road','Share_90']=C2.loc[Avg_share['name']=='road','Share_90']
        Avg_share.loc[Avg_share['name']=='sidewalk','Share_90']=C2.loc[Avg_share['name']=='sidewalk','Share_90']
        Avg_share.loc[Avg_share['name']=='building','Share_90']=C2.loc[Avg_share['name']=='building','Share_90']
        Avg_share.loc[Avg_share['name']=='wall','Share_90']=C2.loc[Avg_share['name']=='wall','Share_90']
        Avg_share.loc[Avg_share['name']=='fence','Share_90']=C2.loc[Avg_share['name']=='fence','Share_90']
        Avg_share.loc[Avg_share['name']=='pole','Share_90']=C2.loc[Avg_share['name']=='pole','Share_90']
        Avg_share.loc[Avg_share['name']=='traffic light','Share_90']=C2.loc[Avg_share['name']=='traffic light','Share_90']
        Avg_share.loc[Avg_share['name']=='traffic sign','Share_90']=C2.loc[Avg_share['name']=='traffic sign','Share_90']
        Avg_share.loc[Avg_share['name']=='vegetation','Share_90']=C2.loc[Avg_share['name']=='vegetation','Share_90']
        Avg_share.loc[Avg_share['name']=='terrain','Share_90']=C2.loc[Avg_share['name']=='terrain','Share_90']
        Avg_share.loc[Avg_share['name']=='sky','Share_90']=C2.loc[Avg_share['name']=='sky','Share_90']
        Avg_share.loc[Avg_share['name']=='rider','Share_90']=C2.loc[Avg_share['name']=='rider','Share_90']
        Avg_share.loc[Avg_share['name']=='car','Share_90']=C2.loc[Avg_share['name']=='car','Share_90']
        Avg_share.loc[Avg_share['name']=='truck','Share_90']=C2.loc[Avg_share['name']=='truck','Share_90']
        Avg_share.loc[Avg_share['name']=='bus','Share_90']=C2.loc[Avg_share['name']=='bus','Share_90']
        Avg_share.loc[Avg_share['name']=='train','Share_90']=C2.loc[Avg_share['name']=='train','Share_90']
        Avg_share.loc[Avg_share['name']=='motorcycle','Share_90']=C2.loc[Avg_share['name']=='motorcycle','Share_90']
        Avg_share.loc[Avg_share['name']=='bicycle','Share_90']=C2.loc[Avg_share['name']=='bicycle','Share_90']
        Avg_share.loc[Avg_share['name']=='license plate','Share_90']=C2.loc[Avg_share['name']=='license plate','Share_90']
     
    
    #angle 180
    angle=180
    base_filename='Pic_'+name+'_Angle_'+str(angle)
    suffix = '.jpg'
    pic_path = os.path.join(Folder_directory, base_filename + suffix)
    
    if os.path.isfile(pic_path):
        img = Image.open(pic_path)
        labels = model.predict_one(img)
        unique, counts= np.unique(labels, return_counts=True)
        data_tuples = list(zip(unique, counts))
        list(Label.columns)
        A= pd.DataFrame(data_tuples, columns=['trainId','Frequency'])
        A['Share_180']=A['Frequency']/sum(A['Frequency'])
        C3= B.merge(A, how='left', on='trainId')
        C3= C3[['trainId','name','Share_180']]
        
        Avg_share.loc[Avg_share['name']=='Unlabeled','Share_180']=C3.loc[Avg_share['name']=='Unlabeled','Share_180']
        Avg_share.loc[Avg_share['name']=='person','Share_180']=C3.loc[Avg_share['name']=='person','Share_180']
        Avg_share.loc[Avg_share['name']=='road','Share_180']=C3.loc[Avg_share['name']=='road','Share_180']
        Avg_share.loc[Avg_share['name']=='sidewalk','Share_180']=C3.loc[Avg_share['name']=='sidewalk','Share_180']
        Avg_share.loc[Avg_share['name']=='building','Share_180']=C3.loc[Avg_share['name']=='building','Share_180']
        Avg_share.loc[Avg_share['name']=='wall','Share_180']=C3.loc[Avg_share['name']=='wall','Share_180']
        Avg_share.loc[Avg_share['name']=='fence','Share_180']=C3.loc[Avg_share['name']=='fence','Share_180']
        Avg_share.loc[Avg_share['name']=='pole','Share_180']=C3.loc[Avg_share['name']=='pole','Share_180']
        Avg_share.loc[Avg_share['name']=='traffic light','Share_180']=C3.loc[Avg_share['name']=='traffic light','Share_180']
        Avg_share.loc[Avg_share['name']=='traffic sign','Share_180']=C3.loc[Avg_share['name']=='traffic sign','Share_180']
        Avg_share.loc[Avg_share['name']=='vegetation','Share_180']=C3.loc[Avg_share['name']=='vegetation','Share_180']
        Avg_share.loc[Avg_share['name']=='terrain','Share_180']=C3.loc[Avg_share['name']=='terrain','Share_180']
        Avg_share.loc[Avg_share['name']=='sky','Share_180']=C3.loc[Avg_share['name']=='sky','Share_180']
        Avg_share.loc[Avg_share['name']=='rider','Share_180']=C3.loc[Avg_share['name']=='rider','Share_180']
        Avg_share.loc[Avg_share['name']=='car','Share_180']=C3.loc[Avg_share['name']=='car','Share_180']
        Avg_share.loc[Avg_share['name']=='truck','Share_180']=C3.loc[Avg_share['name']=='truck','Share_180']
        Avg_share.loc[Avg_share['name']=='bus','Share_180']=C3.loc[Avg_share['name']=='bus','Share_180']
        Avg_share.loc[Avg_share['name']=='train','Share_180']=C3.loc[Avg_share['name']=='train','Share_180']
        Avg_share.loc[Avg_share['name']=='motorcycle','Share_180']=C3.loc[Avg_share['name']=='motorcycle','Share_180']
        Avg_share.loc[Avg_share['name']=='bicycle','Share_180']=C3.loc[Avg_share['name']=='bicycle','Share_180']
        Avg_share.loc[Avg_share['name']=='license plate','Share_180']=C3.loc[Avg_share['name']=='license plate','Share_180']
     
    #angle 360
    angle=270
    base_filename='Pic_'+name+'_Angle_'+str(angle)
    suffix = '.jpg'
    pic_path = os.path.join(Folder_directory, base_filename + suffix)
    if os.path.isfile(pic_path):
        img = Image.open(pic_path)
        labels = model.predict_one(img)
        unique, counts= np.unique(labels, return_counts=True)
        data_tuples = list(zip(unique, counts))
        list(Label.columns)
        A= pd.DataFrame(data_tuples, columns=['trainId','Frequency'])
        A['Share_270']=A['Frequency']/sum(A['Frequency'])
        C4= B.merge(A, how='left', on='trainId')
        C4= C4[['trainId','name','Share_270']]
        
        Avg_share.loc[Avg_share['name']=='Unlabeled','Share_270']=C4.loc[Avg_share['name']=='Unlabeled','Share_270']
        Avg_share.loc[Avg_share['name']=='person','Share_270']=C4.loc[Avg_share['name']=='person','Share_270']
        Avg_share.loc[Avg_share['name']=='road','Share_270']=C4.loc[Avg_share['name']=='road','Share_270']
        Avg_share.loc[Avg_share['name']=='sidewalk','Share_270']=C4.loc[Avg_share['name']=='sidewalk','Share_270']
        Avg_share.loc[Avg_share['name']=='building','Share_270']=C4.loc[Avg_share['name']=='building','Share_270']
        Avg_share.loc[Avg_share['name']=='wall','Share_270']=C4.loc[Avg_share['name']=='wall','Share_270']
        Avg_share.loc[Avg_share['name']=='fence','Share_270']=C4.loc[Avg_share['name']=='fence','Share_270']
        Avg_share.loc[Avg_share['name']=='pole','Share_270']=C4.loc[Avg_share['name']=='pole','Share_270']
        Avg_share.loc[Avg_share['name']=='traffic light','Share_270']=C4.loc[Avg_share['name']=='traffic light','Share_270']
        Avg_share.loc[Avg_share['name']=='traffic sign','Share_270']=C4.loc[Avg_share['name']=='traffic sign','Share_270']
        Avg_share.loc[Avg_share['name']=='vegetation','Share_270']=C4.loc[Avg_share['name']=='vegetation','Share_270']
        Avg_share.loc[Avg_share['name']=='terrain','Share_270']=C4.loc[Avg_share['name']=='terrain','Share_270']
        Avg_share.loc[Avg_share['name']=='sky','Share_270']=C4.loc[Avg_share['name']=='sky','Share_270']
        Avg_share.loc[Avg_share['name']=='rider','Share_270']=C4.loc[Avg_share['name']=='rider','Share_270']
        Avg_share.loc[Avg_share['name']=='car','Share_270']=C4.loc[Avg_share['name']=='car','Share_270']
        Avg_share.loc[Avg_share['name']=='truck','Share_270']=C4.loc[Avg_share['name']=='truck','Share_270']
        Avg_share.loc[Avg_share['name']=='bus','Share_270']=C4.loc[Avg_share['name']=='bus','Share_270']
        Avg_share.loc[Avg_share['name']=='train','Share_270']=C4.loc[Avg_share['name']=='train','Share_270']
        Avg_share.loc[Avg_share['name']=='motorcycle','Share_270']=C4.loc[Avg_share['name']=='motorcycle','Share_270']
        Avg_share.loc[Avg_share['name']=='bicycle','Share_270']=C4.loc[Avg_share['name']=='bicycle','Share_270']
        Avg_share.loc[Avg_share['name']=='license plate','Share_270']=C4.loc[Avg_share['name']=='license plate','Share_270']
    
    Avg_share['average_share']=Avg_share.loc[:, ["Share_0","Share_90","Share_180","Share_270"]].mean(axis=1)
    
    
    Processed_Data['unlabeled'][n]=Avg_share['average_share'].loc[Avg_share['name']=='Unlabeled'].item()
    Processed_Data['road'][n]=Avg_share['average_share'].loc[Avg_share['name']=='road'].item()
    Processed_Data['sidewalk'][n]=Avg_share['average_share'].loc[Avg_share['name']=='sidewalk'].item()
    Processed_Data['building'][n]=Avg_share['average_share'].loc[Avg_share['name']=='building'].item()
    Processed_Data['wall'][n]=Avg_share['average_share'].loc[Avg_share['name']=='wall'].item()
    Processed_Data['fence'][n]=Avg_share['average_share'].loc[Avg_share['name']=='fence'].item()
    Processed_Data['pole'][n]=Avg_share['average_share'].loc[Avg_share['name']=='pole'].item()
    Processed_Data['traffic light'][n]=Avg_share['average_share'].loc[Avg_share['name']=='traffic light'].item()
    Processed_Data['traffic sign'][n]=Avg_share['average_share'].loc[Avg_share['name']=='traffic sign'].item()
    Processed_Data['vegetation'][n]=Avg_share['average_share'].loc[Avg_share['name']=='vegetation'].item()
    Processed_Data['terrain'][n]=Avg_share['average_share'].loc[Avg_share['name']=='terrain'].item()
    Processed_Data['sky'][n]=Avg_share['average_share'].loc[Avg_share['name']=='sky'].item()
    Processed_Data['person'][n]=Avg_share['average_share'].loc[Avg_share['name']=='person'].item()
    Processed_Data['rider'][n]=Avg_share['average_share'].loc[Avg_share['name']=='rider'].item()
    Processed_Data['car'][n]=Avg_share['average_share'].loc[Avg_share['name']=='car'].item()
    Processed_Data['truck'][n]=Avg_share['average_share'].loc[Avg_share['name']=='truck'].item()
    Processed_Data['bus'][n]=Avg_share['average_share'].loc[Avg_share['name']=='bus'].item()
    Processed_Data['train'][n]=Avg_share['average_share'].loc[Avg_share['name']=='train'].item()
    Processed_Data['motorcycle'][n]=Avg_share['average_share'].loc[Avg_share['name']=='motorcycle'].item()
    Processed_Data['bicycle'][n]=Avg_share['average_share'].loc[Avg_share['name']=='bicycle'].item()
    Processed_Data['license plate'][n]=Avg_share['average_share'].loc[Avg_share['name']=='license plate'].item()
    print(n)
drop_colnames=['unlabeled' ,'road', 'sidewalk','building', 'wall','fence', 
       'pole','traffic light','traffic sign','vegetation', 'terrain', 'sky',
       'person', 'rider', 'car', 'truck', 'bus','train','motorcycle','bicycle','license plate']

Processed_Data.to_csv('Processed_Data_'+places+'.csv')

df=Processed_Data.dropna(subset=drop_colnames, how='all')

df.to_csv('Processed_Data_clean_'+places+'.csv')

#df=pd.read_csv('UtrechtZwolle/Processed_Data_clean.csv',sep=',',dtype={0:'str',1:'str',2:'str'})
#df=df.drop(df.columns[0], axis=1)
df['P']=df.sum(axis=1)
df2=df
df2.iloc[:,2:]=df2.iloc[:,2:].div(df.P, axis=0)

df2.to_csv('Processed_Data_clean_standarized_'+places+'.csv')

import os
import osmnx as ox
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum, multidict, exp
import numpy as np
from math import radians, cos, sin, asin, sqrt
import time
import random
import itertools
import pandas as pd
import inequalipy as ineq
import geopandas as gpd
from shapely.geometry import Point 
from shapely import wkt
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


'''
APPROXIMATE DISTANCES BETWEEN LAT/LON POINTS
'''

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

'''
GET DATA
'''
def get_data(place_name,fire_origin):
    area = ox.geocode_to_gdf(place_name)
    area_1 = ox.geocode_to_gdf("Paradise, CA")
    city = area.plot(fc="goldenrod", ec="darkgoldenrod")
    city_1=area_1.plot(fc="goldenrod", ec="darkgoldenrod")

    #building gets all buildings, tried 'building':'residential' to get only residential, but some of the info is missing for smaller cities.
    tags_orig = {'building': True}

    #amenity gets all amentities, such as school. School specifies only schools. Can also do churches, etc.
    tags_dests = {'amenity': 'school'}   
   
    #gets all origins within 1000 meters of center point
    origins_points=ox.geometries_from_point(fire_origin, tags_orig, dist=5000)    
    
    #gets polygon of all schools
    dests_points = ox.geometries_from_place(place_name, tags_dests)
    
    #plot origins as points in red on the city axis
    origins_points.plot(ax=city_1, color="darkred", label="residential buildings")
    #plot dests as points in green over the city
    #dests_points.plot(ax=city_1,color="darkgreen", label="school buildings")
    
    city_1.set_title('Building data from OSMnx', fontdict={"fontsize": "14", "fontweight" : "3"})
    #locate legend for map
    city_1.legend(loc='upper left', framealpha=0 )
    #turn off lat long axis numbers
    city_1.axis("off")
    
    #get centroids of building footprints
    dests_points['geometry'] = dests_points['geometry'].centroid
    #get centroid points and make them the geometry of the gdf
    origins_points['geometry'] = origins_points['geometry'].centroid
    
    #this saves to data frame that only contains name and geometry, none of the other stuff 
    dests_points  = dests_points.loc[:,dests_points.columns.str.contains('geometry')]
    
    #get all destinations that are not safe (here that is any destination within 15 km)
    not_dests=ox.geometries_from_point(fire_origin,tags_dests,dist=15000) 

    #get centroids
    not_dests['geometry'] = not_dests['geometry'].centroid
    #this saves to data frame that only contains name and geometry, none of the other stuff 
    not_dests = not_dests.loc[:,not_dests.columns.str.contains('geometry')]
    
    #remove all impacted destinations from destinations list
    for i in not_dests.index:
        if i in dests_points.index:
            dests_points=dests_points.drop(i)
      
    #remove all impacted destinations from destinations list       
    for i in origins_points.index:
        if i in dests_points.index:
            dests_points=dests_points.drop(i)
      
   
    
    #plot origins as points in red on the city axis
    origins_points.plot(ax=city, marker='x',color="darkred",  markersize=10, label="impacted residential buildings")
    #plot dests as points in green over the city
    dests_points.plot(ax=city,color="darkgreen", markersize=10,label="safe potential shelters")
    #plot unsafe dests
    not_dests.plot(ax=city,color="red", markersize=10,label="unfasfe shelters")
    #add title to map
    city.set_title('Potential Evacuation Shelters', fontdict={"fontsize": "14", "fontweight" : "3"})
    #locate legend for map
    city.legend(loc='upper left', framealpha=0 )
    #turn off lat long axis numbers
    city.axis("off")

    #this saves to data frame that only contains name and geometry, none of the other stuff 
    dests  = dests_points.loc[:,dests_points.columns.str.contains('geometry')]
    orig  = origins_points.loc[:,origins_points.columns.str.contains('geometry')]

    #make separate lat and lon columns for each dataframe (for other program)
    orig['lon'] = orig['geometry'].x
    orig['lat'] = orig['geometry'].y
    dests['lon'] = dests['geometry'].x
    dests['lat'] = dests['geometry'].y

    #compute distances between each pair of origins and destinations
    distances =[]
    populations=[]
    for o,row_orig in orig.iterrows():
        populations.append([o[1],1])
        for d,row_dest in dests.iterrows():
	    #approximates distance between two points in km
            distance= haversine(row_orig['lon'],row_orig['lat'],row_dest['lon'],row_dest['lat'])
            distances.append([o[1], d[1], distance])
		
    #convert to dataframe
    distances = pd.DataFrame(distances, columns=['origins','destinations','distances'])
    populations = pd.DataFrame(populations, columns=['origins','populations'])

    #make multi index
    distances.set_index(['origins','destinations'], inplace=True)
    populations.set_index(['origins'], inplace=True)

    return(orig,dests,distances,populations)

'''
calculate aversion to inequality parameter
'''
def calculate_kappa(origins,destinations,distances,populations):

    #randomly open 5 destinations to calulate aversion to inequality 
    open_current = random.sample(destinations,1)
    
    distances_open = distances.loc[pd.IndexSlice[:, open_current], :]

    nearest = distances_open.groupby('origins')['distances'].min().values

    
    # calculate the alpha value
    kappa = ineq.kolmpollak.calc_kappa(nearest, epsilon = epsilon, weights=populations['populations'].tolist())
    alpha = -kappa
    
    logger.info('kappa is calculated: '+ str(kappa))
    
    return alpha
    
'''
Run optimization to minimize number of shelters to open with every resident having access within a given threshold
'''
    
def optimize_kpcoef(origins, destinations, populations, distances, distances_copy, alpha, kpcoef):

    model = Model()
    
    logger.info('set variables')
    # x_d is binary, 1 if destination d is opened, 0 otherwise
    x = {d: model.addVar(vtype="B") for d in destinations}
    
    # y_o,d is binary, 1 if destination d is assigned to origin o, 0 otherwise
    y = {i: model.addVar(vtype="B") for i in itertools.product(origins, destinations)}

    logger.info('set constraints')
    # constraint: each origin can only be assigned a single destination
    for o in origins:
        model.addCons(quicksum(y[o, d] for d in destinations) == 1)

    # constraint: an origin cannot be assigned an unopen destination
    for d in destinations:
        for o in origins:
            model.addCons(y[o, d]-x[d] <= 0)
            
    # Kolm-Pollak Constraint
    model.addCons(quicksum(populations['populations'][o]*y[o,d]*distances['distances'][o,d] for d in destinations for o in origins) - kpcoef <= 0)
    
    # NEW objective: minimize the number of destinations
    logger.info('set objective')
    model.setObjective(quicksum(x[d] for d in destinations), 'minimize')

    # solve the model
    logger.info('optimizing')
    model.optimize()
    logger.info('optimization complete')
    
    # identify which facilities are opened (i.e., their value = 1)
    new_facilities = np.where([int(round(model.getVal(x[d]))) for d in destinations])[0]
    new_facilities_list=[]
    for new in new_facilities:
        new_location = destinations_list[new]
        new_facilities_list.append(new_location)
        
    time=model.getSolvingTime()
    av_dist=quicksum(populations['populations'][o]*distances_copy['distances'][o,d]*model.getVal(y[o,d]) for o in origins for d in destinations)
    objective=model.getPrimalbound()
    kp_value=quicksum(populations['populations'][o]*model.getVal(y[o,d])*distances['distances'][o,d] for d in destinations for o in origins)

    return(new_facilities_list,time,av_dist,objective,kp_value)
    
 
'''
Run optimization model to minimize distance traveled for each resident to an evac shelter given number of shelters to open
'''   
def optimize_kpobj(origins, destinations, populations, distances, distances_copy, alpha, num_to_open):

    model = Model()
    
    logger.info('set variables')
    # x_d is binary, 1 if destination d is opened, 0 otherwise
    x = {d: model.addVar(vtype="B") for d in destinations}
    
    # y_o,d is binary, 1 if destination d is assigned to origin o, 0 otherwise
    y = {i: model.addVar(vtype="B") for i in itertools.product(origins, destinations)}

    logger.info('set constraints')
    # constraint: each origin can only be assigned a single destination
    for o in origins:
        model.addCons(quicksum(y[o, d] for d in destinations) == 1)
    
    # constraint: the sum of open destinations should equal the number we want to be open
    model.addCons(quicksum(x[d] for d in destinations) == num_to_open)

    # constraint: an origin cannot be assigned an unopen destination
    for d in destinations:
        for o in origins:
            model.addCons(y[o, d]-x[d] <= 0)
            
    # Kolm-Pollak Constraint
    model.setObjective(quicksum(populations['populations'][o]*y[o,d]*distances['distances'][o,d] for d in destinations for o in origins), 'minimize')

    # solve the model
    logger.info('optimizing')
    model.optimize()
    logger.info('optimization complete')
    
    # identify which facilities are opened (i.e., their value = 1)
    new_facilities = np.where([int(round(model.getVal(x[d]))) for d in destinations])[0]
    new_facilities_list=[]
    for new in new_facilities:
        new_location = destinations[new]
        new_facilities_list.append(new_location)
        
    time=model.getSolvingTime()
    av_dist=quicksum(populations['populations'][o]*distances_copy['distances'][o,d]*model.getVal(y[o,d]) for o in origins for d in destinations)
    kp_value=quicksum(populations['populations'][o]*model.getVal(y[o,d])*distances['distances'][o,d] for d in destinations for o in origins)

    return(new_facilities_list,time,av_dist,num_to_open,kp_value)


'''
Input parameters
'''

#put in the large place that has both impacted and safe areas, i.e. county 
place_name= "Butte County, CA, USA"
#initial fire point  
fire_origin=(39.79302,-121.58823)

#number of evac shelters to open (for kpobj)
num_to_open = 4

#distance in km. Maximum distance threshold approximately 50km. (somewhat arbitrary currently)
kpcoef=25

#get the data
origins,destinations,distances,populations = get_data(place_name,fire_origin)

#add unique destinations to list
destinations_list = destinations.index.get_level_values(1).unique().tolist()

#add unique origins to list
origins_list = populations.index.get_level_values(0).unique().tolist()

#typical value for epsilon (which is part of aversion to inequality parameter)
epsilon=-1

#calculate aversion parameter
alpha=calculate_kappa(origins_list,destinations_list,distances,populations)

#turn kpcoef into kolm pollak distance
kpcoef=len(origins_list)*np.exp(alpha*kpcoef)

#this changes the distances to be e^(distance*alpha)
distances_copy=distances.copy()
distances=[]
for d,row_distance in distances_copy.iterrows():
    dist=np.exp(alpha*row_distance['distances'])
    distances.append([d[0],d[1], dist])

#convert to dataframe
distances = pd.DataFrame(distances, columns=['origins','destinations','distances'])

#make multi index
distances.set_index(['origins','destinations'], inplace=True)

#sorted
distances.sort_index(level=[0,1], inplace=True)
 
#run kpcoef optimization formulation
#open_optimal_kpcoef,kp_coef_time,num_stores,av_dist,kp_value = optimize_kpcoef(origins_list,destinations_list, populations, distances,distances_copy, alpha, kpcoef)

#run kpobj
open_optimal_kpcoef,kp_coef_time,num_stores,av_dist,kp_value = optimize_kpobj(origins_list,destinations_list, populations, distances,distances_copy, alpha, num_to_open)

#get only optimal destinations
dests_kpcoef = destinations.loc[(destinations.index.get_level_values(1)).isin(open_optimal_kpcoef)]


#plot dests on teh city map axis
area_1 = ox.geocode_to_gdf(place_name)
city_1 = area_1.plot(fc="goldenrod", ec="darkgoldenrod")
origins.plot(ax=city_1, marker='x', color='darkred', markersize=20, label="impacted areas")
dests_kpcoef.plot(ax=city_1, marker='o', color='darkgreen', markersize=30, label="evacuation shelters")
#destinations.plot(ax=city_1, marker='o', color='blue', markersize=20, label="potential")

#add title to map
city_1.set_title('Optimal Evacuation Shelters', fontdict={"fontsize": "14", "fontweight" : "3"})

#locate legend for map
city_1.legend(loc='upper left', framealpha=0 )

#turn off lat long axis numbers
city_1.axis("off")

#show it
plt.show()
#save it
plt.savefig('optimizing_sinks.png')


#output sinks to csv
output_path = 'optimal_sinks.csv'
dests_kpcoef.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

#output sinks to csv
output_path = 'sources.csv'
origins.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

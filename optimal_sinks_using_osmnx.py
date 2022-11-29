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
def get_data(place_name,place_name_origins,place_name_destinations):
    area = ox.geocode_to_gdf(place_name)
    city = area.plot(fc="goldenrod", ec="none")

    #building gets all buildings, tried 'building':'residential' to get only residential, but some of the info is missing for smaller cities.
    tags_orig = {'building': True}

    #amenity gets all amentities, such as school. School specifies only schools. Can also do churches, etc.
    tags_dests = {'amenity': 'school'}   

    #gets polygon of all schools
    dests = ox.geometries_from_place(place_name_destinations, tags_dests)

    #make copy of dests
    dests_points=dests.copy()
    #get centroid of each polygon
    dests_points['geometry'] = dests_points['geometry'].centroid
    #not really sure what .shape does, but they use it in all the examples made by the creator of osmnx (need to look into this)
    dests_points.shape
    #plot dests as points in red over the city
    dests_points.plot(ax=city,color="green",label="destinations")

    #gets polygon of all buildings
    orig = ox.geometries_from_place(place_name_origins, tags_orig)

    #initial fire point  
    fire_origin=(39.79302,-121.58823)

    #gets all origins within 1000 meters of center point
    origins=ox.geometries_from_point(fire_origin, tags_orig, dist=1000)
	#make a copty to get centroids
    origin_points=origins.copy()
    #get centroid points and make them the geometry of the gdf
    origin_points['geometry'] = origin_points['geometry'].centroid
    #shape the gdf
    origin_points.shape
    #plot origins on the city axis
    origin_points.plot(ax=city,color="red", label="origins")

    #this saves to data frame that only contains name and geometry, none of the other stuff 
    dests  = dests_points.loc[:,dests_points.columns.str.contains('geometry')]
    orig  = origin_points.loc[:,origin_points.columns.str.contains('geometry')]

    #make separate lat and lon columns for each dataframe
    orig['lon'] = orig['geometry'].x
    orig['lat'] = orig['geometry'].y
    dests['lon'] = dests['geometry'].x
    dests['lat'] = dests['geometry'].y

    #plt.show()
    #plt.close()

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
Need to fix input of location to match input for this program!
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
Input parameters
'''

#put in the large place that has both impacted and safe areas, i.e. county 
place_name= "Butte County, CA, USA"
#safe areas (city contained in county )
place_name_destinations= "Chico, CA, USA"
#impacted area (city countained in county, should be different city than impacted, otherwise could get nodes that are impacted as destinations)
place_name_origins="Paradise, CA, USA"

#get the data
origins,destinations,distances,populations=get_data(place_name,place_name_origins,place_name_destinations)

#add unique destinations to list
destinations_list = destinations.index.get_level_values(1).unique().tolist()
#add unique origins to list
origins_list = populations.index.get_level_values(0).unique().tolist()

#typical value for epsilon (which is part of aversion to inequality parameter)
epsilon=-1

#distance in km. Maximum distance threshold approximately 50km. (somewhat arbitrary currently)
kpcoef=50

#calculate aversion parameter
alpha=calculate_kappa(origins_list,destinations_list,distances,populations)

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
#sort
distances.sort_index(level=[0,1], inplace=True)
 
#run optimization formulation
open_optimal_kpcoef,kp_coef_time,num_stores,av_dist,kp_value = optimize_kpcoef(origins_list,destinations_list, populations, distances,distances_copy, alpha, kpcoef)

#get only optimal destinations
dests_kpcoef =   destinations.loc[(destinations.index.get_level_values(1)).isin(open_optimal_kpcoef)]

#create geometry feature for data frame
geometry = [Point(xy) for xy in zip(dests_kpcoef['lon'], dests_kpcoef['lat'])]
#convert to geodataframe
dests_kpcoef = gpd.GeoDataFrame(dests_kpcoef, geometry=geometry,crs='epsg:4326')

#plot dests on teh city map axis
area = ox.geocode_to_gdf(place_name)
city = area.plot(fc="goldenrod", ec="none")
origins.plot(ax=city, marker='x', color='red', markersize=20, label="impacted areas")
dests_kpcoef.plot(ax=city, marker='o', color='green', markersize=30, label="evacuation shelters")
plt.show()

#dests_kpcoef,origins_kpcoef=map_shelters(city_name)

#output sinks to csv
output_path = 'optimal_sinks.csv'
dests_kpcoef.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

#output sinks to csv
output_path = 'sources.csv'
origins.to_csv(output_path, mode='a', header=not os.path.exists(output_path))


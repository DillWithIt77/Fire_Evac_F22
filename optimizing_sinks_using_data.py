import os
import matplotlib.pyplot as plt
from pyscipopt import Model, quicksum, multidict, exp
import numpy as np
import math
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


def import_files(city_name,impacted_census_tract):
    file_path = './new_data/'

    # populations as a df with origins as index
    logger.info('importing the population')
   
    populations_df = pd.read_csv(file_path + city_name + '-population.csv')
    
    #map id_origin to string in order to get origins that are in the census tract
    populations_df["id_orig_string"] = populations_df["id_orig"].map(str)
    
    #get all impacted nodes and make those the origins
    df_impacted=populations_df.loc[populations_df["id_orig_string"].str.contains(impacted_census_tract)]
    populations=df_impacted
    
    print('populations',populations)
    
    # keep only populated locations for origins
    populations = populations[populations['U7B001'] > 0]
    populations.rename(columns={'id_orig': 'origin', 'U7B001':'population'}, inplace=True)  
    print('populations',populations)
    
    #reindex origins
    populations = populations[['population', 'origin']]
    populations.set_index(['origin'], inplace=True)
    print('populations',populations)
   
    # destinations as a list
    logger.info('importing the destinations')
    destinations_df = pd.read_csv(file_path + city_name + '-destinations.csv')
    print('dests',destinations_df)
    
    
    #map id_dest to string so we can make sure to delete all impacted nodes from potential location list
    destinations_df["id_dest_string"] = destinations_df["id_dest"].map(str)
    print('dests',destinations_df)
    
    #get rid of all impacted nodes as potential locations
    df_safe=destinations_df.loc[~destinations_df["id_dest_string"].str.contains(impacted_census_tract)]
    print('safe',df_safe)
    
    destinations_df=df_safe
    print('dests',destinations_df)
    
    #add unique destinations to list
    destinations = destinations_df.loc[destinations_df['dest_type'].isin(['bg_centroid']), 'id_dest'].unique().tolist()
    print('dests',destinations)
    
    # distances as a df with multi-index: ['origin','destination']
    logger.info('importing the distances')
    distances = pd.read_csv(file_path + city_name + '-distances.csv')
    
    #get distances for only neccesary combinations of origins and destinations, i.e ones that are populated
    distances = distances.loc[distances['id_orig'].isin(populations.index)]
    distances = distances.loc[distances['id_dest'].isin(destinations)]
    
    #rename columns
    distances.rename(columns={'id_orig': 'origin', 'id_dest': 'destination','network':'distance'}, inplace=True)
    distances = distances[['origin', 'destination', 'distance']]
    
    #set indices
    distances.set_index(['origin', 'destination'], inplace=True)
    
    #sort
    distances.sort_index(level=[0,1], inplace=True)

    # origins as a list
    origins = populations.index.get_level_values(0).unique().tolist()

    
    logger.info('data imported')
    return {'origins': origins, 'destinations': destinations, 'populations': populations, 'distances': distances}


def calculate_kappa(location,epsilon):

    #randomly open 5 destinations to calulate aversion to inequality 
    open_current = random.sample(location['destinations'],5)
    
    # determine the nearest distances
    distances = location['distances']
    distances_open = distances.loc[pd.IndexSlice[:, open_current], :]
    nearest = distances_open.groupby('origin')['distance'].min().values

    # calculate the alpha value
    kappa = ineq.kolmpollak.calc_kappa(nearest, epsilon = epsilon, weights=location['populations'].values)
    alpha = -kappa
    
    logger.info('kappa is calculated: '+ str(kappa))
    
    print('alpha:',alpha)

    return alpha


def set_kpcoef(origins, destinations, populations, distances, distances_copy, alpha, kpcoef):
    print('L=', kpcoef)
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
    model.addCons(quicksum(populations[o]*y[o,d]*distances[o,d] for d in destinations for o in origins) - kpcoef <= 0)
    
    # NEW objective: minimize the number of destinations
    logger.info('set objective')
    model.setObjective(quicksum(x[d] for d in destinations), 'minimize')

    # solve the model
    logger.info('optimizing')
    model.optimize()
    logger.info('optimization complete')
    
    # identify which facilities are opened (i.e., their value = 1)
    new_facilities = np.where([int(round(model.getVal(x[d]))) for d in destinations])[0]
    time=model.getSolvingTime()
    av_dist=quicksum(populations[o]*distances_copy[o,d]*model.getVal(y[o,d]) for o in origins for d in destinations)
    objective=model.getPrimalbound()
    kp_value=quicksum(populations[o]*model.getVal(y[o,d])*distances[o,d] for d in destinations for o in origins)
            

    return(new_facilities,time,av_dist,objective,kp_value)
    
    
def map_shelters(city_name):

    #name optimal solution from output
    opt_kpcoef=open_optimal_kpcoef

    #geographic boundary data file
    df_world = gpd.read_file("./new_data/cb_2020_us_bg_500k.shp")

    #import origins/populations of origins
    populations_df = pd.read_csv(file_path + city_name + '-population.csv')

    #import all potential destinations
    destinations_df = pd.read_csv(file_path + city_name + '-destinations.csv')

    #map id_origin to string in order to get origins that are in the census tract
    populations_df["id_orig_string"] = populations_df["id_orig"].map(str)
    print(populations_df)
 
    #get all impacted nodes and make those the origins
    populations_df=populations_df.loc[populations_df["id_orig_string"].str.contains(impacted_census_tract)]

    print('pop',populations_df.keys(),populations_df)

    state_fp=str(populations_df["id_orig_string"][1][0:2])
    county_fp=str(populations_df["id_orig_string"][1][2:5])
    tract_fp=str(populations_df["id_orig_string"][1][5:11])
    
    print('tract_fp',tract_fp)
    
    #filtering the geographic area of df
    df_state = df_world.loc[df_world['STATEFP'] == state_fp]
    df_county = df_state.loc[df_state['COUNTYFP'] == county_fp]
    df_tract = df_county.loc[df_county['TRACTCE'] == tract_fp]
    
    print(df_tract)
    
    #df_tract['centroid_column'] = df_tract.centroid
    #df_tract = df_tract.set_geometry('centroid_column')


    #get only optimal destinations
    dests_kpcoef =   destinations_df.loc[(destinations_df['id_dest'].index).isin(opt_kpcoef)]
    print(dests_kpcoef)

    #create geometry feature for data frame
    geometry = [Point(xy) for xy in zip(dests_kpcoef['lon'], dests_kpcoef['lat'])]

    #convert to geodataframe
    dests_kpcoef = gpd.GeoDataFrame(dests_kpcoef, geometry=geometry,crs='epsg:4326')

    #current origins list doesn't have geometry component... 
    #origins_kpcoef =  populations_df.loc[populations_df["id_orig_string"].str.contains(impacted_census_tract)]
    #print('origins: ',type(origins_kpcoef),origins_kpcoef.head())
    #geometry = [Point(xy) for xy in zip(origins_kpcoef['lon'], dests_kpcoef['lat'])]
    #origins_kpcoef = gpd.GeoDataFrame(origins_kpcoef, geometry=geometry,crs='epsg:4326')
    #print(origins_kpcoef.head())

    #create axis to plot on using geographic boundary data
    citymap = df_world["geometry"].plot(color='gold', edgecolor='goldenrod',linewidth=1)

    #plot origins on the city map axis
    df_tract.plot(ax=citymap, label="origins")

    #plot dests on teh city map axis
    dests_kpcoef.plot(ax=citymap, marker='x', color='red', markersize=30, label="evacuation shelters")

    #add title to map
    citymap.set_title('Optimal Evacuation Shelters', fontdict={"fontsize": "14", "fontweight" : "3"})

    #locate legend for map
    citymap.legend(loc='upper left', framealpha=0 )

    #set custom boundary for map
    minx, miny, maxx, maxy = df_tract.total_bounds
    xdif=2*(maxx-minx)
    ydif=2*(maxy-miny)
    citymap.set_xlim(minx-xdif, maxx+xdif)
    citymap.set_ylim(miny-ydif, maxy+ydif)

    #turn off lat long axis numbers
    citymap.axis("off")

    plt.show()
    plt.savefig('optimizing_sinks.png')
    plt.close()  

    return (dests_kpcoef,df_tract)
    
    
'''
Input parameters: city, impacted census tract, epsilon, kpcoef
'''

# city names are 0 thru 499 for Tom's data, 462 is chico,ca and 307 is pueblo,co (limited to largest cities currently, but will fix soon)
city_name='26'

#path to data in my computer
file_path = './new_data/'

  
#explanation of how fips are broken down
#block: 	state (2), county (3) , tract (6) , block (4), 	ex: 22 071 000400 2004  
#block group:	state (2), county (3) , tract (6) , blockgroup (1),	ex: 22 051 020205 1

#just chose a random census tract to close. Could do several.
#impacted_census_tract = '006041' #chico
impacted_census_tract = '000100' #pueblo and new orleans


#typical value for epsilon (which is part of aversion to inequality parameter)
epsilon=-1

#distance in meters. Maximum distance threshold approximately 10km.
kpcoef=10000

  
'''
run optimization formulation
'''

logger.info('importing data for ' + city_name)
#use the import_files function to get the origins and destinations 
location = import_files(city_name,impacted_census_tract)

logger.info('calculating kappa')
#use the calculate_kappa function to calculate a value of alpha (-kappa) which is the aversion to inequality parameter
alpha = calculate_kappa(location,epsilon)

#format populations and distances as dictionaries
location['populations'] = location['populations']['population'].to_dict()
location['distances'] = location['distances']['distance'].to_dict()

#rename everything
origins , destinations= location['origins'], location['destinations']
populations, distances = location['populations'], location['distances']

#simplify the work the optimization program has to do by making the distances already computed
distances_copy=distances.copy()
for key in distances.keys():
    distances[key]=np.exp(alpha*distances[key])

#run optimization formulation
open_optimal_kpcoef,kp_coef_time,num_stores,av_dist,kp_value = set_kpcoef(origins,destinations, populations, distances,distances_copy, alpha, kpcoef)

print(open_optimal_kpcoef)

dests_kpcoef,origins_kpcoef=map_shelters(city_name)

#output sinks to csv
output_path = 'optimal_sinks.csv'
dests_kpcoef.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

#output sinks to csv
output_path = 'sources.csv'
origins_kpcoef.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

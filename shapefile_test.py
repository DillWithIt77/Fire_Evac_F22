# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on 10-17 12:16:24 2023

@author: cyclo
"""




import shapefile as shapefile
#This reads in the shapefile
sf = shapefile.Reader("blockgroups.shp")

#this tells us what type of "shape" is in our shapefile
s=sf.shapeType
print(s)

#this gives us the coordinates of the boundary box of our shape file
sh=sf.bbox
print(sh)

# creating the list of lat/long
test_list = sh

 
# Making the lat/long points
long = [test_list[i] for i in range(len(sh)) if i % 2 != 0]
lat = [test_list[i] for i in range(len(sh)) if i % 2 == 0]

# printing result
print("The alternate element list is : " + str(long))
print("The alternate element list is : " + str(lat))









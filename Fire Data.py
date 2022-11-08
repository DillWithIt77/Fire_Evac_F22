# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:16:24 2022

@author: cyclo
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely import wkt
from shapely import geometry 
r=max(1,3,7,2)
long = [2.50, 1.23, 4.02, 3.25, 5.00, 4.40]
lat = [34, 62, 49, 22, 13, 19]
p1 = geometry.Point(10,20)
p2 = geometry.Point(10,10)
p3 = geometry.Point(20,10)

pointList = [p1, p2, p3, p1]
poly = geometry.Polygon([[p.x, p.y] for p in pointList])
#p1 = wkt.loads("Polygon((1.23 1.32,1.45 1.29,1.86 1.92))")
#print(p1.centroid.wkt)
fig, ax = plt.subplots()
ax.scatter(long, lat)
cir = plt.Circle((20, 20), r, color='r',fill=False)
ax.set_aspect('equal', adjustable='datalim')
ax.add_patch(cir)
print (poly.wkt)
plt.show()





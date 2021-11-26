import pandas as pd
from classe_function import *
from math import cos, dist, log2, pi, sin
from shapely.geometry import Point
from shapely.geometry import LineString
from matplotlib import pyplot as plt
from shapely.geometry import multilinestring 
from shapely.geometry.multilinestring import MultiLineString
import numpy as np

#Function to plot in PLT

###########################################################################################
###########################################################################################
#A MODIFIER POUR DIM > 2
###########################################################################################
###########################################################################################

def plot_vector(lines, color) : # a faire pour n> 2
    for line in lines.geoms :
        x,y = line.xy
        plt.quiver(x[0],y[0], x[1]-x[0], y[1]-y[0], units='xy', color=color, scale=1, zorder = 2)

# Function to compute the average vector given a set of linestrings
def average_direction_vector(lines):
    X,Y = 0,0
    for line in lines.geoms : #For each line we compute the vector and add it to the total for each X, Y
        x,y = line.xy
        u = x[1] - x[0] 
        v = y[1] - y[0] 
        X += u
        Y += v 
    x_u = X/len(lines.geoms)
    y_u = Y/len(lines.geoms)
    return(x_u, y_u)

# Function to get the angle omega 0 between average vector and the unit vector
# Dot product : u.v = ||u|| * ||v|| * cos 0
# 0 = arccos(u.v / ||u|| * ||v||) 
  
def get_omega(x,y) : # x,y of the vector
    vector = np.array([x,y])
    unit = np.array([1,0])
    dot = np.dot(vector,unit)

    vector_norm = np.linalg.norm(vector)
    unit_norm = np.linalg.norm(unit) #useless cause it's 1

    omega = np.arccos(dot/(vector_norm * unit_norm)) #en radian
    #omega = omega * 180 / pi #en degree

    return(omega) 


#Fonction qui prend x,y en entree et le converti selon la rotation 0
def rotation(x,y, omega) :
    x_prime = cos(omega)*x + sin(omega) * y
    y_prime = -sin(omega)*x + cos(omega)*y

    return(x_prime,y_prime)

# Function qui undo une rotation par omega
def undo_rotation(x_prime, y_prime, omega) : 
    x = cos(omega) * x_prime - sin(omega) * y_prime
    y = sin(omega) * x_prime + cos(omega) * y_prime

    return(x,y)

def rotate_back(line, omega) :

    L = []
    for i in range(len(line.coords)) :
        x_prime = line.coords[i][0]
        y_prime = line.coords[i][1]

        x,y = undo_rotation(x_prime, y_prime, omega)
        point = Point(x,y)
        L.append(point)
    rpz_trajectory = LineString(L)

    return(rpz_trajectory)


###########################################################################################
###########################################################################################
#A MODIFIER POUR DIM > 2
###########################################################################################
###########################################################################################

#Fonction qui prend lines en entree et le converti selon rotation 0
def rotate(lines, omega) :
    L = []
    for line in lines.geoms : # Amodifier pour que ca marche pour dim > 2
        x,y = line.xy
        x_prime_1, y_prime_1 = rotation(x[0],y[0], omega)
        x_prime_2, y_prime_2 = rotation(x[1],y[1], omega)
        u_prime = LineString([(x_prime_1, y_prime_1), (x_prime_2, y_prime_2)])
        L.append(u_prime)
    lines_prime = MultiLineString(L)

    return(lines_prime)

#Fonction qui sort une MultiLineString en foction du premier x de chaque LineString
def sorting(lines) :
    L = []
    i = 0
    global_minima = 0

    for line in lines.geoms : 
        x,y = line.xy
        local_minima = min(x)
        if local_minima < global_minima :
            global_minima = local_minima
    
    for line in lines.geoms : 
        if i == 0 :
            L.append(line)
        else :
            x,y = line.xy
            y = i
            while (x[0] + abs(global_minima)) < (L[y-1].xy[0][0] + abs(global_minima)) :
                if y == 0 :
                    L.insert(y, line)
                    break
                y -= 1
            L.insert(y, line)
        i += 1
    lines_sorted = MultiLineString(L)

    return(lines_sorted)

# Fonction qui recupere les points d'un multiline string
def get_x_y(lines) : 
    X,Y = np.array([]), np.array([])
    for line in lines.geoms : 
        x,y = line.xy
        X = np.append(X,x)
        Y = np.append(Y,y)
    return(X,Y)


# Fonction qui cree une LineString verticale et return le y moyen des points, none si inferieur a
# minLines
def intersection(lines, x_sweep, y_max, y_min, minLines) : 
    L = LineString([(x_sweep,y_min), (x_sweep,y_max)])
    inter = L.intersection(lines)

    try :
        len(inter.geoms)
        if len(inter.geoms) >= minLines :
            mean_y = 0
            for object in inter.geoms :
                y = object.xy[1]
                mean_y += y[0]
            mean_y = mean_y / len(inter.geoms)
            return(mean_y)

    except (TypeError, AttributeError) :
        return(None)

    return(None)

def sweep(lines, minLines, smoothing) :
    # STEP 1 : compute the avg direction vector -- DONE
    # STEP 2 : Rotation -- DONE 
    # STEP 3 (HERE) : Sweep line
    lines_sorted = sorting(lines)
    x_sweep, y_max_sweep = get_x_y(lines_sorted)
    x_sweep.sort()
    y_min_sweep = min(y_max_sweep) - 1
    y_max_sweep = max(y_max_sweep) + 1

    L = []

    for i in range(len(x_sweep)) :
        mean_y = intersection(lines, x_sweep[i], y_max_sweep, y_min_sweep, minLines)
        if mean_y == None :
            continue
        else : 
            if i > 0 and (abs(x_sweep[i-1]) - abs(x_sweep[i]) < smoothing) :
                continue
            #on veut une line string
            P = Point(x_sweep[i], mean_y) #coordonees 0mega
            L.append(P)
    if len(L) <= 1 : 
        return(None)
    else :
        rpz_trajectory = LineString(L)
        return(rpz_trajectory)



def plot_trajectory(line, color) :
    x,y = line.xy
    plt.plot(x,y, color=color, zorder = 3)

def plot_trajectories(lines, color) :
    for line in lines.geoms :
        x,y = line.xy
        plt.plot(x,y, color = color, zorder = 3)




###########################################################################################
###########################################################################################
#ROTATION BACK TO NORMAL X Y
###########################################################################################
###########################################################################################

# from 1 linestring to mutiple linestrings in one multilinestring
def to_multi(line) : 
    L = []
    i = 0
    while i < (len(line.coords) - 1) :
        if i == 0 : 
            segment = LineString([line.coords[0], line.coords[1]])
            L.append(segment)
            i += 1
        else : 
            segment = LineString([line.coords[i], line.coords[i+1]])
            L.append(segment)
            i += 1
    lines = MultiLineString(L)
    return(lines)
    


# ##Vectors definition
# u1 = LineString([(1,1), (7,2)])
# u2 = LineString([(2,2), (5,3)])
# u3 = LineString([(1,3), (5,5)])
# u4 = LineString([(5,4), (10,5)])
# u5 = LineString([(8,3), (12,5)])
# u6 = LineString([(8,2), (11,3)])
# u7 = LineString([(8,2), (8,3), (10,5), (5,5), (11,3)])
# lines = MultiLineString([u1,u2,u3,u4,u5,u6])


# #Axes definition ect
# fig = plt.figure()
# ax1 = plt.subplot(1,2,1)

# plot_vector(lines, 'black')
# x,y = average_direction_vector(lines)
# plt.quiver(4, 2, x, y, units ='xy', color='pink', scale =1, zorder = 2)
# ax1.set_aspect('equal')
# ax1.set_xticks(np.arange(14))
# ax1.set_yticks(np.arange(9))
# plt.grid()
# plt.xlabel('Cluster and its average direction vector')

# omega = get_omega(x,y)
# print("L'angle omega a pour valeur : " + str(omega) + ' radians')
# lines_prime = rotate(lines, omega)

# ax2 = plt.subplot(1,2,2)

# plot_vector(lines_prime, 'green')
# x,y = average_direction_vector(lines_prime)
# plt.quiver(0, 0, x, y, units ='xy', color='orange', scale =1, zorder = 2)

# rpz_trajectory= sweep(lines_prime, 3, 0.5)
# plot_trajectory(rpz_trajectory, 'red')

# #ax2.vlines(x_sweep, -1,8, color='black') #sweep
# plt.grid()
# ax2.set_aspect('equal')
# ax2.set_xticks(np.arange(14))
# ax2.set_yticks(np.arange(9))
# plt.xlabel('Same cluster rotated by Omega')

# plt.show()



# f, ax = plt.subplots()
# u7_bis = to_multi(u7)
# plot_vector(u7_bis, 'blue')

# ax.set_aspect('equal')
# ax.set_xticks(np.arange(14))
# ax.set_yticks(np.arange(9))

# plt.grid()

# plt.show()
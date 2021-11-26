from math import dist, atan2, sin, log2
import numpy as np
from shapely.geometry import Point, multilinestring
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


# Prend une ligne et la decompose en plusieurs segements
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
    
def plot_vector(lines, color) : # a faire pour n> 2
    for line in lines.geoms :
        x,y = line.xy
        plt.quiver(x[0],y[0], x[1]-x[0], y[1]-y[0], units='xy', color=color, scale=1, zorder = 2)


class cluster:
    def __init__(self, clusterID, color, type) : 
        self.clusterID = clusterID
        self.color = color
        self.type = 'NONE'

# Projection of point P onto the segment [u,v] (lineString)

def projection(u, v, p) : 
    point = Point(p)
    line = LineString([u, v])

    x = np.array(point.coords[0])
    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])

    n = v - u
    n /= np.linalg.norm(n, 2)

    P = u + n*np.dot(x - u, n)
    P = Point(P)
    return P

# Extracting coordinates from the 2 linestrings and projetting lj onto li

def extract_coord(line_i, line_j) : 
    si, ei = Point(np.array(line_i.coords[0])), Point(np.array(line_i.coords[len(line_i.coords)-1]))
    sj, ej = Point(np.array(line_j.coords[0])), Point(np.array(line_j.coords[len(line_j.coords)-1]))

    ps = projection(si, ei, sj) #ps = projection de sj sur Li[si,ei]
    pe = projection(si, ei, ej) #pe = projection de ej sur Li[si,ei]

    return si, ei, sj, ej, ps, pe

# Perpendicular distance
def perpendicular_distance(line_i, line_j) :
    si, ei, sj, ej, ps, pe = extract_coord(line_i, line_j)

    l_perp_1 = sj.distance(ps)
    l_perp_2 = ej.distance(pe)

    try : 
        r = ((l_perp_1 ** 2) + (l_perp_2 ** 2)) / (l_perp_1 + l_perp_2)
    except (ZeroDivisionError) : 
        r = 0 
    return r

# Parallel Distance
def parallel_distance(line_i, line_j) :
    si, ei, sj, ej, ps, pe = extract_coord(line_i, line_j)

    l_par_1 = min(ps.distance(si), ps.distance(ei))
    l_par_2 = min(pe.distance(si), pe.distance(ei))

    r = min(l_par_1, l_par_2)
    return r

# Angle Distance
def angle_distance(line_i, line_j) : 
    si, ei, sj, ej, ps, pe = extract_coord(line_i, line_j)

    alpha_1 = atan2(ei.y - si.y, ei.x - si.x)
    alpha_2 = atan2(ej.y - sj.y, ej.x - sj.x)

    d = sj.distance(ej)

    r = abs(d * sin(alpha_2 - alpha_1))
    return r

# Final Distance function between the 2 lines
def final_distance(line_i, line_j) : 

    T = perpendicular_distance(line_i,line_j)
    P = parallel_distance(line_i, line_j)
    O = angle_distance(line_i, line_j)

    ####### Optimizing these weights through ML/DL ? ########
    wT, wP, wO = 1,1,1

    D = wT * T + wP * P + wO * O
    return D

# L(H)
def LH_Partition(line) : 
    r = log2(Point(np.array(line.coords[0])).distance(Point(np.array(line.coords[len(line.coords)-1]))))
    return r

# L(D|H)
def LDH_Partition(line) : 
    s, e = Point(np.array(line.coords[0])), Point(np.array(line.coords[len(line.coords)-1]))
    d1 = s.distance(e)

    i,y = 0,1
    total_T_distance, total_O_distance = 0,0

    while(True) : 
        si, ei = Point(np.array(line.coords[i])), Point(np.array(line.coords[y]))
        line_i = LineString([si,ei])
        d2 = si.distance(ei)

        if (d1 >= d2) : 
            total_T_distance += perpendicular_distance(line, line_i)
            total_O_distance += angle_distance(line, line_i)
        else :
            total_T_distance += perpendicular_distance(line_i, line) #inversion car on va alors projeter d1 sur d2
            total_O_distance += angle_distance(line_i, line)

        if y == (len(line.coords)-1) :
            break

        i += 1
        y += 1
    if total_O_distance == 0 :
        r = log2(total_T_distance)
        return r
    r = log2(total_T_distance) + log2(total_O_distance)
    return r

# Cost function
def cost(line) : 
    LH = LH_Partition(line)
    LDH = LDH_Partition(line)

    cost = LH + LDH
    return cost

#Partitionnement, on remplace line par trajectory pour faciliter la comprehension (mais ca reste une LineString)
#input : trajectory
#output : set CP of charasteristic points
# local optimum is the LONGEST trajectory partition that satisfies MDLpar <= MDLno par
def Partition(trajectory) : 
    CP = [Point(trajectory.coords[0])] # add p1 into CP (starting point)
    start_index = 0
    length = 2
    
    while (start_index+length) <= (len(trajectory.coords)-1) :
        curr_index = start_index + length
        start_point, end_point = Point(np.array(trajectory.coords[start_index])), Point(np.array(trajectory.coords[curr_index]))
        L = [start_point, Point(np.array(trajectory.coords[start_index +1]))]
        L.append(end_point)
        line_i = LineString(L)
        cost_1 = cost(line_i) #MDL partition (p start to p curr index) (MDL par = LH+LDH)
        cost_2 = LH_Partition(line_i)#MDL no partition (pstart to p curr index) (MDL no par = LH car LDH = 0)
        if cost_1 > cost_2 :
            CP.append(Point(trajectory.coords[curr_index - 1]))
            start_index = curr_index - 1
            length = 2
        else :
            length += 1

    CP.append(Point(trajectory.coords[len(trajectory.coords) -1]))
    line = LineString(CP)
    return(line)

#epsilon neighborhood
def epsilon_neighborhood(line, lines, epsilon) : #Return a list of indexes of neighbor
    i = 0
    L = []
    while i < len(lines.geoms) :
        if lines.geoms[i] == line :
            i += 1
            continue
        D = final_distance(line, lines.geoms[i]) 
        if D <= epsilon : 
            L.append(i)
        i += 1
    return(L)

#expand Cluster
def expandCluster(lines, queue, epsilon, minLns, clusterID, Unclassified, Noise, clusters): 

    while len(queue) != 0 : 
        M = queue[0]
        Ne = epsilon_neighborhood(M, lines, epsilon)
        if len(Ne) >= minLns :
            for i in Ne : 
                if lines.geoms[i] in Unclassified :
                    queue.append(lines.geoms[i])
                if (lines.geoms[i] in Unclassified) or (lines.geoms[i] in Noise) :
                    clusters[clusterID].append(lines.geoms[i])
                    Unclassified.remove(lines.geoms[i])
        queue.remove(M)

#grouping
def grouping(lines, epsilon, minLns, clusters, Noise, Unclassified) :
    clusterID = 0
    queue = []
    #barre de progression len(lines.geoms)
    for i in tqdm(range(len(lines.geoms)), file = sys.stdout) : 
        if lines.geoms[i] in Unclassified : 
            Ne = epsilon_neighborhood(lines.geoms[i], lines, epsilon)
            if (len(Ne) + 1) >= minLns : 
                clusters[clusterID].append(lines.geoms[i])
                for y in Ne : 
                    clusters[clusterID].append(lines.geoms[y])
                    queue.append(lines.geoms[y])
                expandCluster(lines, queue, epsilon, minLns, clusterID, Unclassified, Noise, clusters)
                clusterID += 1
            else :
                Noise.append(lines.geoms[i])

#last step
def cardinality(clusters, minClusters, lines, Removed) :
    for i in clusters : 
        if len(i) < minClusters :
            indexes = i
            clusters.remove(i)
            for y in indexes :
                Removed.append(lines[y])



def plot_coords(ax, ob, color):
    for line in ob:
        x,y = line.xy
        plt.scatter(x,y, color= color, figure = ax, zorder = 1)

def plot_lines(ax, ob, color):
    for line in ob:
        x,y = line.xy
        plt.plot(x,y, color= color, figure = ax, zorder = 2)



###################################################################################
################### TESTING #######################################################
###################################################################################

'''
line_i = LineString([(0,0), (3,0)])
line_j = LineString([(1,1),(3,3)])

T = perpendicular_distance(line_i,line_j)
P = parallel_distance(line_i, line_j)
O = angle_distance(line_i, line_j)

print('Perpendicular distance : ' + str(T))
print('Parallel distance : ' + str(P))
print('Angle distance : ' + str(O))

wT, wP, wO = 1,1,1

D = wT * T + wP * P + wO * O

print('Final Distance : ' + str(D))
'''
#############
# traj = LineString([(1,1), (1,2), (1.2,3.1),(1.3,3.2), (2.9,2.9), (3,3.3), (3.1,4), (3.4,5)])
# C = Partition(traj)
# C = to_multi(C)

# f,ax = plt.subplots()
# plot_vector(C,'black')

# x,y = traj.xy
# plt.plot(x,y, color='green')

# ax.set_aspect('equal')
# ax.set_xticks(np.arange(5))
# ax.set_yticks(np.arange(7))
# plt.grid()


# plt.show()

'''
line = LineString([(2,3), (3,3)])
line2 = LineString([Point(1,1), Point(1.25,3)])
lines = multilinestring(line, line2)
lines2 = multilinestring(line, line2)

D = final_distance(line, line2)

print("Final Distance")
print(D)

print("LineString Distance")
print(lines.distance(lines2))

x2,y2 = line.xy
plt.plot(x2,y2, color='red')
x,y = line2.xy
plt.plot(x,y, color='green')
plt.show()


line = LineString([(0,0), (1,1)])
line2 = LineString([(0,2), (1,1.5), (1.5,1), (2,0)])
line3 = LineString([(6,3), (5,4)])
line4 = LineString([(3,4), (5,5.5), (1,7), (3,6)])
line5 = LineString([(0,1), (2,3)])
lines = MultiLineString([line, line2, line3, line4, line5])

Removed = []
Noise = []
Unclassified = []
clusters= [[],[],[],[],[],[],[],[],[],[],[]]

for i in lines.geoms : 
    Unclassified.append(i)

print(Unclassified)

grouping(lines, 2, 2, clusters, Noise, Unclassified)
print(clusters)
'''

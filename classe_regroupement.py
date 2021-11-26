from classe_autres_tests import *
from classe_function import *
import csv
import  pandas as pd
import math
from matplotlib.pyplot import cm
from time import sleep
from tqdm import tqdm 
import matplotlib


Removed = []
Noise = []
Unclassified = []
clusters= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]


def plot_vector2(lines, color) : # a faire pour n> 2
    head_length = 0.2
    for line in lines.geoms :
        x,y = line.xy
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        vec_ab_magnitude = math.sqrt(dx**2+dy**2)

        dx = dx / vec_ab_magnitude
        dy = dy / vec_ab_magnitude

        vec_ab_magnitude = vec_ab_magnitude - head_length

        ax.arrow(x[0], y[0], vec_ab_magnitude*dx, vec_ab_magnitude*dy, head_width=0.2, head_length=head_length, fc=color, ec=color, zorder = 3)

# Function that takes a list of points in this format : "0 1 4 5" for '(0,1), (4,5)'
# from this list, return a line string
def to_line(list) :
    L = []
    i = 0
    while i < len(list) :
        p = Point(list[i], list[i+1])
        L.append(p)
        i += 2
    line = LineString(L)
    return(line)

def input_trajectory(input) : 
    # Step 1 : Open the csv
    with open(input, mode = 'r', newline = '') as traj_csv :
        csv_reader = csv.reader(traj_csv)
        rows = []
        for row in csv_reader :
            rows.append(row)
    traj_csv.close()

    # Step 2 : convert the strings number to int
    for each in rows :
        for i in range(len(each)) :
            each[i] = int(each[i])

    # Step 3 : for each row of the csv doc, convert it to a LineString
    # Then, segmentation of this line string
    # Finally, take the segmentation in a multilinestring of different segments
    # Returns L, a List of MultiLineString
    L = []
    for row in rows : 
        line = to_line(row)
        C = Partition(line)
        C = to_multi(C)
        L.append(C)
    return(L)

trajectories_segmented = input_trajectory('Trajectories_Building.csv')

L = []
for i in trajectories_segmented : 
    for y in i.geoms : 
        L.append(y)

def get_trajectories(clusters, minLines, smoothing) :
    L = []
    for cluster in clusters : 
        lines = MultiLineString(cluster)
        x,y = average_direction_vector(lines)
        omega = get_omega(x,y)
        lines_prime = rotate(lines, omega)
        trajectory = sweep(lines_prime, minLines, smoothing)
        try : 
            trajectory = rotate_back(trajectory, omega)
            L.append(trajectory)
        except (AttributeError) :
            continue
    trajectories = MultiLineString(L)

    return(trajectories)

##############################################################################
##############################################################################
# Lines = MultiLineString contenant chaque segment
lines = MultiLineString(L)



# Clustering : 
# Step 1 : Chaque segment = 'Unclassified' : deja fait inherent a la classe

# Step 2 : Clustering
nb_clusters = grouping(lines, 1, 3, clusters, Noise, Unclassified)

fig,ax = plt.subplots()

#while len(clusters[-1]) == 0 : 
#    clusters.pop()

color= iter(cm.gnuplot(np.linspace(0,1, nb_clusters + 1)))
#color= iter(cm.gnuplot(np.linspace(0,1,len(clusters))))
for index in range(nb_clusters + 1) : 
#for index in range(len(clusters)) : 
    c = next(color)
#    for line in clusters[index] :
    for line in lines :
        if line.clusterID == index :
            x,y = line.xy
            plt.plot(x,y, color = c, alpha = 0.2)

batiment = matplotlib.patches.Rectangle((5,6), 8, 5, color = 'dimgrey', alpha = 1, fill = False, linewidth = 2)
ascenceur1 = matplotlib.patches.Rectangle((4,9), 1, 1, color = 'forestgreen', alpha = 1, fill = True)
ascenceur2 = matplotlib.patches.Rectangle((13,9), 1, 1, color = 'forestgreen', alpha = 1, fill = True)
acceuil = matplotlib.patches.Rectangle((7,5), 4, 1, color = 'cornflowerblue' , alpha = 1, fill = True)
entrance1 = matplotlib.patches.Rectangle((8,0), 2, 1, color = 'orange', fill =True)
entrance2 = matplotlib.patches.Rectangle((18,5), 1, 1, color = 'orange', fill = True)
detente = matplotlib.patches.Rectangle((13,0), 6, 2, color = 'firebrick', fill = True)

ax.add_patch(batiment)
ax.add_patch(ascenceur1)
ax.add_patch(ascenceur2)
ax.add_patch(acceuil)
ax.add_patch(entrance1)
ax.add_patch(entrance2)
ax.add_patch(detente)

ax.set_axisbelow(True)
plt.grid()
ax.set_xticks(np.arange(18))
ax.set_yticks(np.arange(11))
#plt.show()


# trajectories = get_trajectories(clusters, minLines = 2, smoothing = 0)
# plot_trajectories(trajectories, 'red')





# cluster1 = MultiLineString(clusters[1])
# fig2,ax = plt.subplots()
# plot_vector2(cluster1, 'black')
# plt.show()
# x,y = average_direction_vector(cluster1)
# omega = get_omega(x,y)
# cluster1_prime = rotate(cluster1, omega)
# trajectory = sweep(cluster1_prime, minLines = 2, smoothing = 0)
# trajectory = rotate_back(trajectory, omega)

# fig2,ax = plt.subplots()
# plot_trajectory(trajectory, 'red')
# plot_vector2(cluster1, 'black')


#plt.grid()
ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(13))
plt.show()

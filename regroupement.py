from autres_tests import *
from function import *
import csv
import  pandas as pd
import math


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

        ax.arrow(x[0], y[0], vec_ab_magnitude*dx, vec_ab_magnitude*dy, head_width=0.2, head_length=head_length, fc=color, ec=color, zorder = 2)

with open('Trajectories_Building.csv', mode = 'r', newline = '') as traj_csv :
    csv_reader = csv.reader(traj_csv)
    rows = []
    for row in csv_reader :
        rows.append(row)
traj_csv.close()

for each in rows :
    for i in range(len(each)) :
        each[i] = int(each[i])

def to_line(list) :
    L = []
    i = 0
    while i < len(list) :
        p = Point(list[i], list[i+1])
        L.append(p)
        i += 2
    line = LineString(L)
    return(line)


# for row in rows : 
#     line = to_line(row)
#     x,y = line.xy
#     fig, ax = plt.subplots()
#     plt.plot(x,y)

f,ax = plt.subplots()
for row in rows : 
    line = to_line(row)
    C = Partition(line)
    C = to_multi(C)


    plot_vector2(C,'black')

    x,y = line.xy
    plt.plot(x,y, color='green')

ax.set_aspect('equal')
ax.set_xticks(np.arange(20))
ax.set_yticks(np.arange(12))
plt.grid()
plt.show()
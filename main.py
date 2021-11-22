from function import *

def plot_coords(ax, ob, color):
    for line in ob : 
        x,y = line.xy
        plt.plot(x,y, color= color, figure = ax, zorder = 1)

#### Partition effectuee : tests des etapes suivantes 

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

##print(Unclassified)

grouping(lines, 0.5, 1, clusters, Noise, Unclassified)
print(Noise)
print(clusters[0][0])


ax = plt.figure()

plot_coords(ax, clusters[0], 'green')
plot_coords(ax, clusters[1], 'red')
plot_coords(ax, clusters[2], 'orange')

ax2 = plt.figure()
plot_coords(ax2,lines,'blue')

plt.show()



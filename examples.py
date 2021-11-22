from shapely.geometry import Point, LineString
from random import randrange
from matplotlib import pyplot as plt 
from shapely.geometry.multilinestring import MultiLineString

def plot_coords(ax, ob, color):
    for line in ob:
        x,y = line.xy
        plt.plot(x,y, color= color, figure = ax, zorder = 1)

L = []

for i in range(50) :
    x = randrange(1,10)
    y = randrange(1,50)
    c = randrange(1,10)
    u = randrange(1,50)
    p = Point(x,y)
    o = Point(c,u)  
    A = LineString([p,o])

    L.append(A)

X1 = randrange(1,3)
X2 = randrange(7,9)
Y1 = randrange(10,15)
Y2 = randrange(30,40)


LINE = LineString([(X1,Y1), (X2,Y2)])
lines = MultiLineString(L)

print(lines)

ax = plt.figure()
plot_coords(ax, lines, 'green')

q,w = LINE.xy
plt.plot(q,w, color='red', figure = ax)

plt.show()

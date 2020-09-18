import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

xs = np.linspace(0,1,num=10,endpoint=True)
ys = 1*xs

XX,YY = np.meshgrid(xs,ys)

triang = tri.Triangulation(XX.flatten(),YY.flatten())

plt.triplot(triang)

plt.show()

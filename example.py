import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import mpl_toolkits.axisartist as art
import mpl_toolkits.axisartist.grid_helper_curvelinear \
    as grid_helper

from ternary import Ternary
from configureaxes import ConfigureAxes

tern = Ternary()


fig,ax = plt.subplots()


e1,e2 = tern.gen_B1_axes()

f1,f2,border = tern.gen_B2_axes()


def f(x,y):

    ans =  np.exp(-((x-0.2)**2+(y-0.2)**2))*np.cos(4*x)

    return np.where(x+y < 1, ans, np.nan)

x = np.linspace(0,1,num=101,endpoint=True)
y = np.linspace(0,1,num=101,endpoint=True)
XX,YY = np.meshgrid(x,y)

fig, (ax_norm,ax_trans) = plt.subplots(1,2)

ax_norm.contourf(XX,YY,f(XX,YY))


points = tern.B1_to_B2(XX,YY)

ax_trans.contourf(points[0],points[1],f(XX,YY))

tern.turnon_grid(ax_trans)

ConfigureAxes(ax_trans)



fig.savefig("fig2.pdf")

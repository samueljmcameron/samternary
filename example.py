import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ternary import Ternary

tern = Ternary()


fig,ax = plt.subplots()


e1,e2 = tern.gen_e1_e2_axes()

f1,f2,border = tern.gen_f1_f2_axes()


def f(x,y):

    ans =  np.exp(-((x-0.2)**2+(y-0.2)**2))

    return np.where(x+y < 1, ans, np.nan)

x = np.linspace(0,1,num=101,endpoint=True)
y = np.linspace(0,1,num=101,endpoint=True)
XX,YY = np.meshgrid(x,y)

fig,(ax_norm,ax_trans) = plt.subplots(1,2)

ax_norm.contourf(XX,YY,f(XX,YY))


points = tern.f_to_e(XX,YY)
ax_trans.contourf(points[0],points[1],f(XX,YY))




ax.plot(f1[0],f1[1],'k-')
ax.plot(f2[0],f2[1],'k-')
ax.plot(border[0],border[1],'k-')


ax.axis('square')


fig.savefig("fig2.pdf")

import matplotlib.pyplot as plt
import numpy as np

from .configureaxes import ConfigureAxes



def f(x,y):

    ans =  np.exp(-((x-0.2)**2+(y-0.2)**2))*np.cos(4*x*(3-y)**2)
    

    return np.where(x+y <= 1, ans, np.nan)

x = np.linspace(0,1,num=101,endpoint=True)
y = np.linspace(0,1,num=101,endpoint=True)
XX,YY = np.meshgrid(x,y)
fs = f(XX,YY)
    

fig, (ax_norm,ax_trans) = plt.subplots(1,2,
                                       figsize=[5,2.8])


# plot data in normal way first
ax_norm.contourf(XX,YY,f(XX,YY))
ax_norm.set_xlabel(r'$\phi_1$')
ax_norm.set_ylabel(r'$\phi_2$')

# transform ax_trans to ternary-plot style, and load
# in all change of basis features
cob = ConfigureAxes(ax_trans)

# affine transform x,y points to ternary-plot basis
points = cob.B1_to_B2(XX,YY)

cs = ax_trans.contourf(points[0],points[1],fs)

ax_norm.set_title("Cartesian "
                  "(basis " + r"$\mathcal{B}_1$" + ")")
ax_trans.set_title("mesh-grid "
                   "(basis " + r"$\mathcal{B}_2$" + ")")

cbar = fig.colorbar(cs,ax=ax_trans,shrink=0.6)
fig.subplots_adjust(bottom=0.2,hspace=0.01)
plt.show()
fig.savefig("example_meshgrid.pdf")

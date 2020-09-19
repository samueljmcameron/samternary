import matplotlib.pyplot as plt
import numpy as np

from .configureaxes import ConfigureAxes



def f(x,y):

    ans =  np.exp(-((x-0.2)**2+(y-0.2)**2))*np.cos(4*x*(3-y)**2)
    

    return np.where(x+y <= 1, ans, np.nan)

def flatten(XX,YY,fs):

    xr = XX.flatten()
    yr = YY.flatten()
    zr = fs.flatten()

    xr = xr[~np.isnan(zr)]
    yr = yr[~np.isnan(zr)]
    zr = zr[~np.isnan(zr)]
    
    return xr,yr,zr
    
x = np.linspace(0,1,num=101,endpoint=True)
y = np.linspace(0,1,num=101,endpoint=True)
XX,YY = np.meshgrid(x,y)
fs = f(XX,YY)

#flatten data
xr,yr,zr = flatten(XX,YY,fs)

fig, (ax_norm,ax_trans) = plt.subplots(1,2,
                                       figsize=[5,2.8])


# plot data in normal way first

ax_norm.tricontourf(xr,yr,zr)
ax_norm.set_xlabel(r'$\phi_1$')
ax_norm.set_ylabel(r'$\phi_2$')

# transform ax_trans to ternary-plot style, and load
# in all change of basis features
cob = ConfigureAxes(ax_trans)
points = cob.B1_to_B2(xr,yr)

# affine transform x,y points to ternary-plot basis
cs = ax_trans.tricontourf(points[0],points[1],zr)


ax_norm.set_title("Cartesian "
                  "(basis " + r"$\mathcal{B}_1$" + ")")
ax_trans.set_title("flattened-grid "
                   "(basis " + r"$\mathcal{B}_2$" + ")")

cbar = fig.colorbar(cs,ax=ax_trans,shrink=0.6)
fig.subplots_adjust(bottom=0.2,hspace=0.01)
plt.show()
fig.savefig("example_flatarrays.pdf")

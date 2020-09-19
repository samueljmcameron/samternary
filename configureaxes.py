import numpy as np
import matplotlib.tri as tri
from .changeofbasis import ChangeOfBasis

class ConfigureAxes(ChangeOfBasis):
    """
    Set up the axes of the plot to be triangular.
    """

    def __init__(self,ax, gridon=True,
                 grid_lw = '0.5',grid_color='k',
                 grid_ls='--',**kwargs):

        super().__init__()

        self.ax = ax
        self.tick_styles = self.set_kwargs(**kwargs)
        
        self.axdict = self._construct_dict()

        self._draw_ticks()
        self._bound_axes()

        if gridon:
            self._turnon_grid(lw=grid_lw,color=grid_color,
                              ls = grid_ls)

        return


    def set_kwargs(self,**kwargs):

        options = {
            'bottom_name' : r'$\phi_1$',
            'left_name' : r'$\phi_2$',
            'right_name' : r'$\phi_0$',
            'minor_ticklength' : 10,
            'major_ticklength' : 20,
            'bottom_minor_ticklength' : 10,
            'bottom_major_ticklength' : 20,
            'left_minor_ticklength' : 10,
            'left_major_ticklength' : 20,
            'right_minor_ticklength' : 10,
            'right_major_ticklength' : 20,
            'ticklabelpad' : 4,
            'bottom_axis_labelpad' : 18,
            'left_axis_labelpad' : 20,
            'right_axis_labelpad' : 20,
            'bottom_labels_fmt' : '.1f',
            'left_labels_fmt' : '.1f',
            'right_labels_fmt' : '.1f',
            'label_size' : 8,
            'minor_tick_num' : 51,
        }


        lvals = np.linspace(0,1,num=6,endpoint=True)

        for item in ['bottom','left','right']:
            fmt = options[f'{item}_labels_fmt']
            ls = [f'{lval:{fmt}}' for lval in lvals]
            options[f'{item}_labels'] = ls

        
        options.update(kwargs)
        
        return options
    
    def _construct_dict(self):
 
        right_dict = {}
        bottom_dict = {}
        left_dict = {}
        
        num = self.tick_styles['minor_tick_num']

        bottom_ax,left_ax,right_ax = self.gen_B2_axes(num=num)

        bottom_dict['x'] = bottom_ax[0]
        bottom_dict['y'] = bottom_ax[1]
        left_dict['x'] = left_ax[0]
        left_dict['y'] = left_ax[1]
        right_dict['x'] = right_ax[0]
        right_dict['y'] = right_ax[1]

        
        axdict ={}
        axdict['bottom'] = bottom_dict
        axdict['left'] = left_dict
        axdict['right'] = right_dict

        for key,val in axdict.items():

            val['line'], = self.ax.plot(val['x'],
                                        val['y'],
                                        'k-',lw=1)
            

        return axdict

    def _draw_ticks(self):

        lpad = self.tick_styles['ticklabelpad']
        lsize = self.tick_styles['label_size']
        
        minor_num = self.tick_styles['minor_tick_num']
        
        for key,val in self.axdict.items():
            
            ax_lpad = self.tick_styles[f'{key}_'
                                       'axis_labelpad']
            name = self.tick_styles[f'{key}_name']
            min_tl = self.tick_styles[f'{key}_minor'
                                      '_ticklength']
            maj_tl = self.tick_styles[f'{key}_major'
                                      '_ticklength']

            labels = self.tick_styles[f'{key}_labels']

            major_num = len(labels)
            maj_skip = int((minor_num-1)/(major_num-1))
            
            xs = val['x']
            ys = val['y']

            dx,dy = self._compute_normals(xs,ys)        

            for i in range(len(xs)):

                if i % maj_skip == 0:

                    tl = maj_tl

                    self.ax.text(xs[i]+lpad*tl*dx[i],
                                 ys[i]+lpad*tl*dy[i],
                                 labels[i//maj_skip],
                                 ha='center',
                                 va='center',
                                 fontsize=lsize)

                else:
                    tl = min_tl

                xtmp = [xs[i],xs[i]+tl*dx[i]]
                ytmp = [ys[i],ys[i]+tl*dy[i]]
                self.ax.plot(xtmp,ytmp, 'k-',lw=1)

                if i == len(xs)//2:

                    self.ax.text(xs[i]+ax_lpad*tl*dx[i],
                                 ys[i]+ax_lpad*tl*dy[i],
                                 name,
                                 ha='center',
                                 va='center')

        return

    def _turnon_grid(self,lw='0.5',color='k',
                     ls = '--'):

        xsmall = np.linspace(0,1,num=11,endpoint=True)
        ysmall = 1*xsmall

        xsm,ysm = np.meshgrid(xsmall,ysmall)

        smallpoints = self.B1_to_B2(xsm,ysm)

        xflat = smallpoints[0].flatten()
        yflat = smallpoints[1].flatten()

        triang = tri.Triangulation(xflat,yflat)

        mask = yflat[triang.triangles].mean(axis=1)<=0
        triang.set_mask(mask)
        self.ax.triplot(triang,lw=lw,color=color,
                        linestyle=ls)

        return

    
    def _bound_axes(self):

        self.ax.axis('equal')        
        self.ax.set_xlim(-0.1,1.1)
        self.ax.set_ylim(-0.1,np.sqrt(3)/2.0+0.1)
        self.ax.set_axis_off()
        return

    def _get_normal_vec(self,u1, u2,direction):
        """Return the unit vector perpendicular to the
        vector u2-u1, with direction = 1 indicating that
        if u2-u1 points to the right, the normal will
        point up."""

        u1 = np.asarray(u1)
        u2 = np.asarray(u2)

        tangent = (u2-u1)/np.linalg.norm(u2-u1,axis=0)

        normal = np.array([tangent[1],-tangent[0]],float)
        return normal*direction


    def _compute_normals(self,xs,ys):

        u1 = np.asarray((xs[:-2],ys[:-2]))
        u2 = np.asarray((xs[1:-1],ys[1:-1]))
        u3 = np.asarray((xs[2:],ys[2:]))


        x1,y1 = self._get_normal_vec(u1,u2,1)
        x2,y2 = self._get_normal_vec(u2,u3,1)

        xav = np.average([x1,x2],axis=0)
        yav = np.average([y1,y2],axis=0)
        norm = np.sqrt(xav**2+yav**2)

        xav = xav/norm
        yav = yav/norm

        dx = np.concatenate(([x1[0]],xav,[x2[-1]]))*0.001
        dy = np.concatenate(([y1[0]],yav,[y2[-1]]))*0.001

        return dx,dy




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    fig,ax = plt.subplots()

    ConfigureAxes(ax)

    plt.show()

from matplotlib.axis import Tick
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


class ConfigureAxes():

    def __init__(self,ax, **kwargs):
        

        self.ax = ax
        self.tick_styles = self.set_kwargs(**kwargs)
        
        self.axdict = self._construct_dict()

        self._draw_ticks()
        self._bound_axes()

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

        num = self.tick_styles['minor_tick_num']
        ts = np.linspace(0,1,num=num,endpoint=True)
        

        right_dict = {}
        bottom_dict = {}
        left_dict = {}

        bottom_dict['x'] = 1*ts
        bottom_dict['y'] = 0*ts
        left_dict['x'] = ts[::-1]/2.
        left_dict['y'] = np.sqrt(3)*ts[::-1]/2.
        right_dict['x'] = 1-ts/2.
        right_dict['y'] = np.sqrt(3)/2.*(1-ts[::-1])


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
    
    def _bound_axes(self):

        self.ax.set_xlim(-0.1,1.1)
        self.ax.set_ylim(-0.1,np.sqrt(3)/2.0+0.1)
        self.ax.axis('equal')
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


    import plotpub

    figsize = plotpub.PlotPub()
    fig,ax = plt.subplots(figsize=[figsize[1],figsize[1]])

    ConfigureAxes(ax)

    fig.savefig("dum.pdf")

import numpy as np


class Ternary():

    """ change basis from B1 to B2 where

    B1 := {e1,e2}
    and
    B2 := {f1,f2}.

    Here, the basis vectors in B1 are

    e1=(1,0), e2=(0,1)

    and the basis vectors in B2 are

    f1 = (1,tan(pi/12)), f2 = (tan(pi/12),1).

    This effectively maps the system onto a ternary
    phase diagram.

    """

    
    def __init__(self):
        # define change of basis matrices R, Rinv

        # start with mapping to a triangle with
        # f1 not aligned with e1.
        dum = np.array([[1,np.tan(np.pi/12.)],
                      [np.tan(np.pi/12.),1]],
                     float)

        # rotation matrix to rotate coordinates
        # so that f1 is aligned with e1.
        rot = np.array([[np.cos(np.pi/12.),
                         np.sin(np.pi/12.)],
                        [-np.sin(np.pi/12.),
                         np.cos(np.pi/12.)]],
                       float)

        # change of basis matrix from f to e
        self.R = np.dot(rot,dum)
        # and from e to f
        self.Rinv = np.linalg.inv(self.R)
        return

    def e_to_f(self,v_1,v_2):

        return np.array([self.Rinv[0,0]*v_1
                         +self.Rinv[0,1]*v_2,
                         self.Rinv[1,0]*v_1
                         +self.Rinv[1,1]*v_2])

    def f_to_e(self,v_1,v_2):
        
        return np.array([self.R[0,0]*v_1
                         +self.R[0,1]*v_2,
                         self.R[1,0]*v_1
                         +self.R[1,1]*v_2])

    def gen_e1_e2_axes(self):

        e_1x = np.linspace(0,1,num=10,endpoint=True)
        e_1y = 0*e_1x
    
        e_2x = 0*e_1x
        e_2y = 1*e_1x

        e_1 = [e_1x,e_1y]
        e_2 = [e_2x,e_2y]
    
        return e_1, e_2

    def gen_f1_f2_axes(self,fullframe = True):

        e_1,e_2 = self.gen_e1_e2_axes()

        f_1 = self.f_to_e(e_1[0],e_1[1])
        
        f_2 = self.f_to_e(e_2[0],e_2[1])

        if fullframe:
            s_x = 1*e_1[0]
            s_y = 1-s_x
            bord = self.f_to_e(s_x,s_y)

        else:
            bord = None 
        return f_1,f_2,bord





if __name__=="__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns

    tern = Ternary()
    
    
    fig,ax = plt.subplots()
    

    e1,e2 = tern.gen_e1_e2_axes()

    f1,f2,border = tern.gen_f1_f2_axes()
    
    ax.plot(e1[0],e1[1],'k-')
    ax.plot(e2[0],e2[1],'k-')

    ax.plot(f1[0],f1[1],'r-')
    ax.plot(f2[0],f2[1],'r-')
    ax.plot(border[0],border[1],'r-')


    Ae = [0.2,0.2]
    Be = [0.3,0.2]
    Ce = [0.2,0.3]

    points = [Ae,Be,Ce]


    colors = sns.color_palette()

    for i,point in enumerate(points):

        ax.plot(point[0],point[1],'*',color=colors[i])

        pf = tern.f_to_e(point[0],point[1])

        ax.plot(pf[0],pf[1],'s',color=colors[i])


    ax.axis('square')

    fig.savefig("fig.pdf")

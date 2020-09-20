import numpy as np


class ChangeOfBasis():

    """ Affine change of basis from B1 to B2 where

    B1 := {e1,e2}
    and
    B2 := {f1,f2}.

    Here, the basis vectors in B1 are

    e1=(1,0), e2=(0,1)

    and the basis vectors in B2 defined via the
    transformation matrix

    R = [[1,tan(pi/12),0],[-tan(pi/12),-1,1],[0,0,1]].

    So, if writing a vector v in B1 coordinates, i.e.
   
    v = (x,y,1)_{B1},

    the same vector can be written in B2 coordinates,
    i.e.

    v = (u,v,1)_{B2}
    via

    (x,y,1)_{B1} = R^{-1}(u,v,1)_{B2},

    or equivalently

    (u,v,1)_{B2} = R (x,y,1)_{B1}.

    """

    
    def __init__(self):
        # define change of basis matrices R, Rinv

        # start with mapping to a triangle with
        # f1 not aligned with e1.

        #rotation angle
        alpha = np.pi/12.
        # new axis (NOT orthogonal)
        dum1 = np.array([[np.cos(alpha),np.sin(alpha)],
                         [-np.sin(alpha),-np.cos(alpha)]],
                        float)

        rot = np.array([[np.cos(np.pi/12.),
                         np.sin(np.pi/12.)],
                        [-np.sin(np.pi/12.),
                         np.cos(np.pi/12.)]],float)

        dum2 = np.dot(rot,dum1)

        dum = np.array([[dum2[0,0],dum2[0,1],0],
                        [dum2[1,0],dum2[1,1],1],
                        [0,0,1]],float)
        
        #dum = np.array([[1,np.tan(np.pi/12),0],
        #                [-np.tan(np.pi/12),-1,1],
        #                [0,0,1]],float)

        # rotation matrix to rotate coordinates
        # so that f1 is aligned with e1.
        beta = np.pi/6.
        rot = np.array([[np.cos(beta),
                         np.sin(beta),0],
                        [-np.sin(beta),
                         np.cos(beta),0],
                        [0,0,1]], float)

        # change of basis matrix from f to e
        self.R = np.dot(rot,dum)
        # and from e to f
        self.Rinv = np.linalg.inv(self.R)
        return

    def B2_to_B1(self,v_1,v_2):

        v_1, v_2 = np.asarray(v_1), np.asarray(v_2)
        return np.array([self.Rinv[0,0]*v_1
                         +self.Rinv[0,1]*v_2
                         +self.Rinv[0,2],
                         self.Rinv[1,0]*v_1
                         +self.Rinv[1,1]*v_2
                         +self.Rinv[1,2]])

    def B1_to_B2(self,v_1,v_2):
        
        v_1, v_2 = np.asarray(v_1), np.asarray(v_2)
        return np.array([self.R[0,0]*v_1
                         +self.R[0,1]*v_2
                         +self.R[0,2],
                         self.R[1,0]*v_1
                         +self.R[1,1]*v_2
                         +self.R[1,2]])

    def gen_B1_unitvecs(self,num=10):

        e_1x = np.linspace(0,1,num=num,endpoint=True)
        e_1y = 0*e_1x
    
        e_2x = 0*e_1x
        e_2y = 1*e_1x

        e_1 = [e_1x,e_1y]
        e_2 = [e_2x,e_2y]
    
        return e_1, e_2

    def gen_B1_axes(self,num=10):

        return self.gen_B1_unitvecs(num=num)

    def gen_B2_unitvecs(self,num=10,frameon=False):

        e_1,e_2 = self.gen_B1_axes(num=num)

        f_1 = self.B1_to_B2(e_1[0],e_1[1])


        
        f_2 = self.B1_to_B2(e_2[0],e_2[1])

        if frameon == True:
            s_x = 1*e_1[0]
            s_y = 1-s_x
            bord = self.B1_to_B2(s_x,s_y)
        else:
            born = None

        return f_1, f_2, bord

    def gen_B2_axes(self,num=10):

        f_1,f_2,bord = self.gen_B2_unitvecs(num=num,
                                            frameon=True)
        bottom_ax = 1*bord
        left_ax = np.array([f_2[0],f_2[1]])
        right_ax = np.array([f_1[0][::-1],f_1[1][::-1]])

        return bottom_ax,left_ax,right_ax


if __name__=="__main__":

    import matplotlib.pyplot as plt

    cob = ChangeOfBasis()

    
    
    fig,ax = plt.subplots()
    

    e1,e2 = cob.gen_B1_axes()

    f1,f2,border = cob.gen_B2_axes()

    print(f1,f2,border)
    ax.plot(e1[0],e1[1],'k-')
    ax.plot(e2[0],e2[1],'k-')

    ax.plot(f1[0],f1[1],'r-')
    ax.plot(f2[0],f2[1],'g-')
    ax.plot(border[0],border[1],'b-')


    Ae = [0.2,0.2]
    Be = [0.2,0.3]
    Ce = [0.3,0.2]


    points = [Ae,Be,Ce]


    colors = ['r','g','b']

    for i,point in enumerate(points):

        ax.plot(point[0],point[1],'*',color=colors[i])

        pf = cob.B1_to_B2(point[0],point[1])

        ax.plot(pf[0],pf[1],'s',color=colors[i])


    ax.axis('square')

    plt.show()

import numpy as np
from NURBS_C import tensor_product
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time
from scipy.spatial import Delaunay
import winsound

class NURBS():
    points={}
    def __init__(self,point_cloud,num_u,num_w,u_order=3,w_order=3,
                 reshape=False,factor=1,data_dims=[50,50],parse=False):
        '''Used to store the point cloud data and eventually 
        calculate the NURBS surface.
        Takse as initialization:
            point cloud: set of points
            num_u: number of point to evaluate in u direction
            num_v: number of point to evaluate in v direction
            u_order/w_order: order of curve in u/w direction,
                             nominally 4
        '''
        if reshape:
            self._extract_data(point_cloud,data_dims,reshape,factor,parse)
        else:
            self.control_points=point_cloud
        self.u_order=u_order
        self.w_order=w_order
        self.num_u,self.num_w=num_u,num_w
        
    def _extract_data(self,point_cloud,data_dims,reshape,factor,parse):
        '''extract the point cloud data into the x,y, and z values'''
        print('Extracting a portion of the data to fit')
        #if need be pare the data down to a more resaonable number of values
        #to fit the surface to
        x=point_cloud[:,0][::factor]
        y=point_cloud[:,1][::factor]
        z=point_cloud[:,2][::factor]
        #need to pad the values in order to reshape nicely
        control_u=data_dims[0]
        control_w=data_dims[1]
        if parse:
            pad=abs(len(x)-control_u*control_w) if \
                len(x)<control_u*control_w else 0
            
            y_new=np.concatenate([y,y[-pad::]])
            x_new=np.concatenate([x,x[-pad::]])
            z_new=np.concatenate([z,z[-pad::]])
        else:
            y_new,x_new,z_new=y,x,z
        #reshape the arrays to use for control points
        self.control_points=np.array([
            np.reshape(x_new,[control_u,control_w]),
            np.reshape(y_new,[control_u,control_w]),
            np.reshape(z_new,[control_u,control_w])])
        self.points['reduced data']=self.control_points
        
    def _knot(self,k,n,end=True):
            '''Returns the knot vector- with end point interpolation
            optional
                k=order of the curve
                n=number of control vertices
            '''
            print('Developing knot vector')
            if end: #if end point interpolation is desired
                t=np.zeros(k+1)
                for i in range(k+n+1-2*(k+1)):
                    t=np.concatenate([t,np.array([i+1])])
                t=np.concatenate([t,np.array([i+2]*(k+1))])
            else: #if no end point interpolation is needed
                t=np.linspace(0,n+k,n+k+1)
            #normalize t-> this is later undone, but for external uses,
            #it works nicely to have a normalized knot vector
            t=np.array([i/max(t) for i in t])
            return t

    def nurbs_calc(self,weight=[1]):
        #the order of the function 
        ku=self.u_order
        kw=self.w_order
        #get the number of control points in each direction
        #n= num in u and m= num in w
        n,m=np.shape(self.control_points)[1::]
        scale=1
        #generate the u and w vectors
        u=np.linspace(0,n-ku,self.num_u)
        w=np.linspace(0,m-kw,self.num_w)
        if len(weight)==1:
            weight=np.ones([n,m])       
            for i in range(n):
                for j in range(m):
                    weight[i][j]*=(scale*(i**3+1)*(j**0.5+1))
        #unpack the control points for easier use
        xc,yc,zc=self.control_points
        #get the p and w knot vectors here so as to not recalc them
        knot_u,knot_w=self._knot(ku,n)*(n-ku),self._knot(kw,m)*(m-kw)
        #initiate the output 
        output=np.array([np.zeros([self.num_u,self.num_w]),
                np.zeros([self.num_u,self.num_w]),
                np.zeros([self.num_u,self.num_w])])
        #compute the tensor product/get the surface
        print('Beginning to calculate tensor product')
        s=time.time()
        output=np.asarray(tensor_product(xc,yc,zc,weight,u,w,n,m,ku,kw,
                              knot_u,knot_w,output))
        winsound.MessageBeep()
        print('Calculated tensor product in {:.2f}s'.format(time.time()-s))
        self.points['nurbs']=output
        self.output=output
        return self.output,weight
    
    def save_stl(self,file_name):
        print('Generating stl file')
        s=time.time()
        def _writer(vec1,vec2,vec3,surf_norm,f):
            f.write('facet normal {:.7f} {:.7f} {:.7f}\n'.format(*n))
            f.write('outer loop\n')
            f.write('vertex {:.7f} {:.7f} {:.7f}\n'.format(*p1))
            f.write('vertex {:.7f} {:.7f} {:.7f}\n'.format(*p2))
            f.write('vertex {:.7f} {:.7f} {:.7f}\n'.format(*p3))
            f.write('endloop \nendfacet\n')
        if 'output' not in globals():
            print('Must calculate surface first')
        else:
            f=open(file_name,'w')
            f.write('solid\n')
            xc,yc,zc=self.output
            xflat,yflat,zflat=xc.flatten(),yc.flatten(),zc.flatten()
            vectors=np.array([xflat,yflat,zflat]).T
            #compute the tesslation of the points to save:
            vecs=Delaunay(vectors)
            vertice_index=vecs.simplices
            for i in range(vertice_index.shape[0]):
                p=vertice_index[i]
                p1=np.array([xflat[p[0]], yflat[p[0]], zflat[p[0]]])
                p2=np.array([xflat[p[1]], yflat[p[1]], zflat[p[1]]])
                p3=np.array([xflat[p[2]], yflat[p[2]], zflat[p[2]]])
                #get the surface unit normal at the triangle
                cross=np.cross(p2-p1,p3-p1)
                n=cross/np.linalg.norm(cross)
                #sent it to the writer
                _writer(p1,p2,p3,n,f)
            f.close()
        print('Finished generating stl file in {:.2f}s'.format(time.time()-s))
        return vecs,vectors
    
    #a set of plotting tools to plot the data generated
    def plot_surfaces(self,plot_name,fig_num=None,plot_title=None,ax=None):
        point_cloud=self.points[plot_name]
        if fig_num!=None:
            fig=plt.figure(fig_num)
            ax=plt.axes(projection='3d')
        ax.plot_surface(point_cloud[0],point_cloud[1],
                        point_cloud[2],color='m')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(plot_title) if plot_title !=None else ax.set_title('Plot')
        fig.tight_layout()
        return fig,ax
        
    def plot_points(self,plot_name,fig_num=None,plot_title=None,fig=None,ax=None):
        point_cloud=self.points[plot_name]
        if fig_num!=None:
            fig=plt.figure(fig_num)
            ax=plt.axes(projection='3d')
        ax.scatter(point_cloud[0],point_cloud[1],point_cloud[2],color='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(plot_title) if plot_title !=None else ax.set_title('Plot')
        fig.tight_layout()
        return fig,ax
        
if __name__=='__main__':
    f=open('nurbs_test_data.csv','r')
    data=f.readlines()
    f.close()
    x,y,z=[],[],[]
    for i in range(1,len(data)):
        line=data[i].split(sep=',')
        x.append(float(line[0]))
        y.append(float(line[1]))
        z.append(float(line[3]))
    x,y,z=np.array(x),np.array(y),np.array(z)
    num=20
    nurbs=NURBS(np.column_stack([x,y,z]),num,num+10,
                data_dims=[8,8],parse=False,reshape=True)
    output,weight=nurbs.nurbs_calc()
    vecs,points=nurbs.save_stl('tester.stl')
    fig,ax=nurbs.plot_surfaces('nurbs',1,'nurbs')
    nurbs.plot_points('reduced data', plot_title='data',fig=fig,ax=ax)
    fig.savefig('TestData.png')
    plt.show()
    xo,yo,zo=output
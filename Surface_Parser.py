import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

class Surface_Parser():
    stored={}
    def __init__(self,points):
        self.points=points
    def convert_2d(self,z_lim,min_elems=10,factor=1):
        '''Return only the points in the bottom section
        '''
        point_cloud=self.points
        x=point_cloud[:,0][::factor]
        y=point_cloud[:,1][::factor]
        z=point_cloud[:,2][::factor]
        x2d,y2d,z2d=[],[],[]
        i=0
        while i<len(y)-2:
            rowx,rowy,rowz=[],[],[]
            while y[i]-y[i+1]<0 and i<len(y)-2:
                if z[i]<=z_lim:
                    rowy.append(y[i])
                    rowx.append(x[i])
                    rowz.append(z[i])
                i+=1
            i+=1
            if len(rowy)>min_elems:
                y2d.append(rowy)
                x2d.append(rowx)
                z2d.append(rowz)
        self.stored['2d data']=[x2d,y2d,z2d]
        return x2d,y2d,z2d
    
    def interpolation2d(self,adjusted_weight=1e-10,no_adjust_weight=100):
        x,y,z=self.stored['2d data']
        #get the longer row first
        lenth=0
        min_y=0
        max_y=0
        for i in range(len(x)):
            if len(x[i])>lenth:
                lenth=len(x[i])
                if min(y[i])<min_y:
                    min_y=min(y[i])
                if max(y[i])>max_y:
                    max_y=max(y[i]) 
        #initiate a weighting matrix
        weights=np.ones([len(x),lenth])
        self.output=np.array([np.ones([len(x),lenth]),
                         np.ones([len(x),lenth]),
                         np.ones([len(x),lenth])])
        for i in range(len(x)):
            if len(x[i])!=lenth:
                new_y=np.linspace(min_y,max_y,lenth)
                new_z=np.zeros(lenth)
                new_x=[np.average(x[i])]*lenth
                adjusted_points=[]
                for k in range(len(x[i])):
                    closed_index=self._find_closeset_value(new_y, y[i][k])
                    new_y[closed_index]=y[i][k]
                    new_z[closed_index]=z[i][k]
                    adjusted_points.append(closed_index)
                for h in range(len(new_x)):
                    if h in adjusted_points:
                        weights[i][h]=adjusted_weight
                    else:
                        weights[i][h]=no_adjust_weight
                self.write_output(new_x, new_y, new_z, i)
            else:
                self.write_output(x[i], y[i], z[i], i)
                for v in range(len(x[i])):
                    weights[i][v]=no_adjust_weight
        self.stored['adjusted points']=self.output
        return self.output,weights
    
    def write_output(self,x,y,z,loc):
        for j in range(len(x)):
            self.output[0][loc][j]=x[j]
            self.output[1][loc][j]=y[j]
            self.output[2][loc][j]=z[j]
        
    def _find_closeset_value(self,new_y,old_y):
        return np.abs(new_y-old_y).argmin()
    
    def plot_surfaces(self,plot_name,fig_num=None,plot_title=None,ax=None):
        point_cloud=self.stored[plot_name]
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
        point_cloud=self.stored[plot_name]
        if fig_num!=None:
            fig=plt.figure(fig_num)
            ax=plt.axes(projection='3d')
        ax.scatter(point_cloud[:,0],point_cloud[:,1],point_cloud[:,2],color='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(plot_title) if plot_title !=None else ax.set_title('Plot')
        fig.tight_layout()
        return fig,ax
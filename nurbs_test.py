from geomdl import BSpline
from geomdl import exchange
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

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
cntrl=np.column_stack([x,y,z])
cn=[]
for i in cntrl:
    cn.append(list(i))
surf=BSpline.Surface()
surf.degree_u=3
surf.degree_v=3

surf.set_ctrlpts(cn,8,8)
surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 
                     3.0, 4.0, 5.0, 5.0,5.0,5.0]
surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 
                     3.0, 4.0, 5.0, 5.0,5.0,5.0]
surf.delta=0.025
surf.evaluate()
exchange.export_csv(surf, 'test.csv')
#open and read the file to plot it for real
f=open('test.csv')
data=f.readlines()
f.close()
x,y,z=[],[],[]
for i in range(1,len(data)):
    line=data[i].split(sep=',')
    x.append(float(line[0]))
    y.append(float(line[1]))
    z.append(float(line[2]))
    
# x,y,z=np.reshape(x,[40,40]),np.reshape(y,[40,40]),np.reshape(z,[40,40])
# fig_num=1
# plot_title='Test'
# if fig_num!=None:
#     fig=plt.figure(fig_num)
#     ax=plt.axes(projection='3d')
# ax.plot_surface(x,y,z,color='m')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title(plot_title) if plot_title !=None else ax.set_title('Plot')
# fig.tight_layout()
# plt.show()
import numpy as np

def _N(int i,int k, double u, double[:] t):
    cdef double first=0
    cdef double last=0
    if k==0:
        if t[i]<=u and u<t[i+1]:
            return 1.0
        return 0.0
    else:
        if t[i+k]-t[i]>0:
            first=((u-t[i])/(t[i+k]-t[i]))*_N(i,k-1,u,t)
        if t[i+k+1]>t[i+1]:
            last=((t[i+k+1]-u)/(t[i+k+1]-t[i+1]))*_N(i+1,k-1,u,t)
        return first+last

def tensor_product(double[:,:] xc, double[:,:] yc, double[:,:] zc,
                   double[:,:]weights,double[:] u, double[:] w, int n, int m,
                   int u_order, int w_order,
                   double[:] knot_u, double[:] knot_w,
                   double[:,:,:] output):
    cdef int h=0
    cdef int i=0
    cdef int t=0
    cdef int j=0
    cdef int g=0
    cdef int e=0
    cdef double basis_u=0
    cdef double basis_w=0
    cdef double point_sumx=0
    cdef double point_sumy=0
    cdef double point_sumz=0
    cdef double no_point_sum=0
    cdef int lu=len(u)
    cdef int lt=len(w)                
    cdef double[:] x=np.ones(m)
    cdef double[:] y=np.ones(m)
    cdef double[:] z=np.ones(m)
    cdef double basis_results=0
    output=np.asarray(output)
    for h in range(lu):
        print('Iteration {}'.format(h))
        if u[h]==n-u_order:
            #fit the final control point in u direction with a NURBS curve
            xu,yu,zu=np.asarray(xc[-1]),np.asarray(yc[-1]),np.asarray(zc[-1])
            weightu=np.asarray(weights[-1])
            u_curve=np.asarray(BSpline(xu,yu,zu,weightu,np.asarray(w),
                                       np.asarray(knot_w),w_order))
            for g in range(lt):
                output[0][-1][g]=u_curve[0][g]
                output[1][-1][g]=u_curve[1][g]
                output[2][-1][g]=u_curve[2][g]
        else:
            for t in range(lt):
                if w[t]==m-w_order:
                    #fit the final control point in w direction with a NURBS curve
                    xw,yw,zw=np.asarray(xc[:,-1]),np.asarray(yc[:,-1]),np.asarray(zc[:,-1])
                    weightw=np.asarray(weights[:,-1])
                    w_curve=np.asarray(BSpline(xw,yw,zw,weightw,np.asarray(u),
                                               np.asarray(knot_u),u_order))
                    for e in range(lu):
                        output[0][e][-1]=w_curve[0][e]
                        output[1][e][-1]=w_curve[1][e]
                        output[2][e][-1]=w_curve[2][e]
                else:
                    point_sumx,point_sumy,point_sumz=0,0,0
                    no_point_sum=0
                    for i in range(n):
                        #get the basis in the u direction
                        basis_u=_N(i, u_order, u[h], knot_u)
                        for j in range(m):
                            #get the basis in the w direction
                            basis_w=_N(j, w_order, w[t], knot_w)
                            #sum the point weighted and no point ctrl pts
                            basis_results=basis_u*basis_w*weights[i][j]
                            no_point_sum+=basis_results
                            point_sumx+=(basis_results*xc[i][j])
                            point_sumy+=(basis_results*yc[i][j])
                            point_sumz+=(basis_results*zc[i][j])
                    output[0][h][t]=(point_sumx/no_point_sum)
                    output[1][h][t]=(point_sumy/no_point_sum)
                    output[2][h][t]=(point_sumz/no_point_sum)
    return output

def BSpline(double[:] xc, double[:] yc, double[:] zc,double[:] weights,
            double[:] u, double[:] knot,int k):
    cdef int n=len(xc)
    cdef int l=len(u)
    cdef int i=0
    cdef int j=0
    cdef double point_sumx=0
    cdef double point_sumy=0
    cdef double point_sumz=0
    cdef double no_point_sum=0
    output=np.array([np.zeros(l),np.zeros(l),np.zeros(l)])
    for i in range(l):
        if u[i]==n-k:
            output[0][i]=xc[-1]
            output[1][i]=yc[-1]
            output[2][i]=zc[-1]
        else:
            no_point_sum=0
            point_sumx,point_sumy,point_sumz=0,0,0
            for j in range(n):
                basis_u=_N(j, k, u[i], knot)
                no_point_sum+=(basis_u*weights[j])
                point_sumx+=(basis_u*xc[j]*weights[j])
                point_sumy+=(basis_u*yc[j]*weights[j])
                point_sumz+=(basis_u*zc[j]*weights[j])
            output[0][i]=(point_sumx/no_point_sum)
            output[1][i]=(point_sumy/no_point_sum)
            output[2][i]=(point_sumz/no_point_sum)       
    return output
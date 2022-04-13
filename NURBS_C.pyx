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
                   double[:] u, double[:] w, int n, int m,
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
    cdef double weight_sum=0
    cdef double no_point_sum=0
    cdef int lu=len(u)
    cdef int lt=len(w)                
    cdef double[:] x=np.ones(m)
    cdef double[:] y=np.ones(m)
    cdef double[:] z=np.ones(m)
    output=np.asarray(output)
    for h in range(lu):
        print('Iteration {}'.format(h))
        if u[h]==n-u_order:
            xu,yu,zu=np.asarray(xc[-1]),np.asarray(yc[-1]),np.asarray(zc[-1])
            u_curve=np.asarray(BSpline(xu,yu,zu,np.asarray(w),
                                       np.asarray(knot_w),w_order))
            for g in range(lt):
                output[0][-1][g]=u_curve[0][g]
                output[1][-1][g]=u_curve[1][g]
                output[2][-1][g]=u_curve[2][g]
        else:
            for t in range(lt):
                if w[t]==m-w_order:
                    xw,yw,zw=np.asarray(xc[:,-1]),np.asarray(yc[:,-1]),np.asarray(zc[:,-1])
                    w_curve=np.asarray(BSpline(xw,yw,zw,np.asarray(u),
                                               np.asarray(knot_u),u_order))
                    for e in range(lu):
                        output[0][e][-1]=w_curve[0][e]
                        output[1][e][-1]=w_curve[1][e]
                        output[2][e][-1]=w_curve[2][e]
                else:
                    for i in range(n):
                        #get the basis in the u direction
                        basis_u=_N(i, u_order, u[h], knot_u)
                        for j in range(m):
                            #get the basis in the w direction
                            basis_w=_N(j, w_order, w[t], knot_w)
                            output[0][h][t]+=(basis_u*basis_w*xc[i][j])
                            output[1][h][t]+=(basis_u*basis_w*yc[i][j])
                            output[2][h][t]+=(basis_u*basis_w*zc[i][j])
    return output

def BSpline(double[:] xc, double[:] yc, double[:] zc,
            double[:] u, double[:] knot,int k):
    cdef int n=len(xc)
    cdef int l=len(u)
    cdef int i=0
    cdef int j=0
    output=np.array([np.zeros(l),np.zeros(l),np.zeros(l)])
    for i in range(l):
        if u[i]==n-k:
            output[0][i]=xc[-1]
            output[1][i]=yc[-1]
            output[2][i]=zc[-1]
        else:
            for j in range(n):
                basis_u=_N(j, k, u[i], knot)
                output[0][i]+=(basis_u*xc[j])
                output[1][i]+=(basis_u*yc[j])
                output[2][i]+=(basis_u*zc[j])
    return output
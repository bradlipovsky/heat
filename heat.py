import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def heat(t_surf,
         tmax = 1000,
         tmin = 0,
         zmax = 100,
         dTdz = 0.02,
         nz = 39,
         nt = 99,
         alpha = 35,
         accumulation = 0):
    
    '''
    Solves the advection/diffusion equation with mixed temperature/heat flux boundary conditions
    '''
    
    z0=0    
    dz = zmax/(nz+1)
    dt = tmax/nt
    t = np.linspace(tmin,tmax, nt+1)
    z = np.linspace(dz,zmax, nz+1)

    cfl = alpha*dt/(dz**2)
    Azz = np.diag([1+2*cfl] * (nz+1)) + np.diag([-cfl] * (nz),k=1)\
        + np.diag([-cfl] * (nz),k=-1)

    w = - accumulation * np.ones(nz)
    abc = w*dt/(2*dz)
    Az = np.diag(abc,k=1) - np.diag(abc,k=-1)
    Az[0,:] =0
    Az[-1,:]=0
    A = Azz - Az


    # Neumann boundary condition
    A[nz,nz-1] = -2*cfl
    b= np.zeros((nz+1,1))
    b[nz] =  2*cfl*dz * dTdz

    # Initial condition: gradient equal to basal gradient and equal to surface temp.
    U=np.zeros((nz+1,nt+1))
    U[:,0] = t_surf[0] + z*dTdz

    for k in range(nt):
        b[0] = cfl*t_surf[k]    #  Dirichlet boundary condition

        c = U[:,k] + b.flatten()
        U[:,k+1] = np.linalg.solve(A,c)

    return U,t,z

def heat_plot(t,t_surf,start_year,end_year,z,U,plot_start_frac=0.95):
    nt=len(t)
    plt.subplots(2,2,figsize=(8,9))
    plt.subplot(221)
    plt.plot(t,t_surf)
    plt.xlim([start_year,end_year])
    plt.grid()
    plt.title('a. Surface temperature forcing')

    plt.subplot(224)
    strt = int((nt+1)*plot_start_frac)
#     for i in range( strt,nt):
#         plt.plot((U[:,i]),z,label=f't={t[i]:.2f}')
    plt.plot((U[:,-1]),z,'-k',linewidth=3,label=f't={t[-3]:.2f}')

    plt.ylim([max(z),0])
    plt.legend()
    plt.title('c. Temperature profile at last time step')
    


    plt.subplot(223)
    c=plt.pcolormesh(t,z,U)
    plt.colorbar(c,location='bottom')
    plt.ylim([max(z),0])
#     plt.xlabel('Calendar year')
    plt.title('b. Temperature distribution through time')

    plt.tight_layout()
    plt.show()
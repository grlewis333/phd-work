import matplotlib.pyplot as plt                 # For normal plotting
from mpl_toolkits.mplot3d import proj3d         # For 3D plotting
import numpy as np                             # For maths
from scipy import ndimage                       # For image rotations
import RegTomoReconMulti as rtr                 # Modified version of Rob's CS code
from scipy import optimize                      # For function minimization
import copy                                     # For deepcopy
try:
    import astra                                    # For tomography framework
    import transforms3d                             # For some rotation work
except:
    print('Astra import failed')

from scipy import constants  
import matplotlib.patches as patches
import matplotlib
from matplotlib.colors import ListedColormap

def generate_tri_pris(n = 100, size_n = 1,pi=1):
    """ 
    Generate triangular prism data (with missing slice)
    
    Input:
    n = number of nodes in each dimension (nxnxn grid)
    size_n = length in nm of each node
    
    Output:
    X,Y,Z,MX,MY,MZ = Gridded coordinates, gridded magnetisation
    """
    
    # Define gradient/intercept of bounding lines
    m1, c1 = 5, 100
    m2, c2 = 0, -25
    m3, c3 = -0.6, 0
    
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    # Assign density
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if y < (m1*x+c1) and y > (m2*x + c2) and y < (m3*x + c3) and ((z >-20 and z<-10) or (z>0 and z<40)):
                    p = pi
                    data.append([x,y,z,p])
                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def generate_sphere(n = 100, size_n = 1,pi=1,c=(0,0,0),r=30):
    """ Generate sphere of radius r centred at c
    """
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

#     c = (0,0,0)
#     r = 30
    
    # Assign density to sphere
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if (x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2 < r**2:
                    p = 1
                    data.append([x,y,z,p])

                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def generate_tetrapod(n = 100, size_n = 1,pi=1, r_tet=40,r_cyl = 10):
    """ Generate a tetrapod centred at (0,0,0), A-D labelled vertices,
    starting at top and going c/w. AOB is in the xz plane.
    Length of each leg is r_tet and radius of each leg is r_cyl """

    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    # Tetrahedron with O at centre, A-D labelled vertices starting at top and going c/w. AOB is in the xz plane.
    r_tet = r_tet # length of each leg of the tetrapod (i.e. length of OA, OB, OC, OD)
    c = (0,0,0) # origin of tetrapod - note changing this currently doesn't work...
    h = r_tet * (2/3)**.5 / (3/8)**.5 # height in z of the tetrapod
    
    # Calculate tetrahedron vertices
#     A = (c[0],c[1],c[2]+r_tet)
#     B = (c[0]+(r_tet**2-(h-r_tet)**2)**.5,c[1],c[2]-(h-r_tet))
#     mrot = multi_axis.rotation_matrix(0,0,120)
#     C = np.dot(mrot,B)
#     mrot = multi_axis.rotation_matrix(0,0,-120)
#     D = np.dot(mrot,B)

    # Cylinder from centre to top vertex of the tetrahedron
    r_cyl = r_cyl

    # Assign density to first cylinder
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if (x-c[0])**2 + (y-c[1])**2 < r_cyl**2 and ((z >c[2] and z<(c[2]+r_tet))):
                    p = pi
                    data.append([x,y,z,p])
                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)
    P = P.reshape(len(xs),len(ys),len(zs))

    # Rotate cylinder to get other legs
    OA = rotate_bulk(P,0,0,0)
    OB = rotate_bulk(OA,0,120,0)
    OC = rotate_bulk(OB,0,0,120)
    OD = rotate_bulk(OB,0,0,-120)

    # Add all together and clip between 0 and assigned density
    tetrapod = np.clip((OA + OB + OC + OD),0,pi)
    
    return X,Y,Z,tetrapod

def generate_pillar_cavities(n = 100, size_n = 1,pi=1,x_len=70,y_len=50,z_len=50,r_cyl=15,depth=25,nx=1,ny=1):
    """ Generate box of dimensions (x_len,y_len,z_len) with hollow pillars of 'depth' length
        etched into the top z face. There will be an array of nx x ny pillars, each of radius r_cyl.
    """
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    # Box dimensions
#     x_len = 70
#     y_len = 50
#     z_len = 50

#     # tubes
#     r_cyl = 4
#     depth = 25
#     nx = 5
#     ny = 3
    cs = []

    cxs = np.linspace(-x_len/2,x_len/2,num=nx+2)[1:-1]
    cys = np.linspace(-y_len/2,y_len/2,num=ny+2)[1:-1]

    for cx in cxs:
        for cy in cys:
            cs.append((cx,cy))

    # Assign density to box
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if -x_len/2 < x < x_len/2 and -y_len/2 < y < y_len/2 and -z_len/2 < z < z_len/2:
                    p = 1

                    for c in cs:
                        if (x-c[0])**2 + (y-c[1])**2 < r_cyl**2 and z > z_len/2-depth:
                            p = 0

                    data.append([x,y,z,p])

                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def generate_layered_rod(n = 100, size_n = 1,pi=1,r=25,length=80,disc_width = 10):
    """ Generate cylindrical rod of length 80, with alternating discs every 'disc_width'
    Rod is aligned along z
    """
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    c = (0,0,0)

    # Assign density to rod
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if (y-c[1])**2 + (x-c[0])**2 < r**2 and -length/2 < z < length/2 :
                    p = 1
                    if np.floor(abs(-length/2-z)/disc_width)%2 == 0:
                        p = .25
                        
                    data.append([x,y,z,p])
                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def plot_2d(X,Y,Z,P,s=5,size=0.1, width = 0.005, title='',ax=None,fig=None):
    """
    Plot magnetisation data in 2D
    
    Input:
    x,y = 'Projected' 2D coordinates (nxn)
    u,v = 'Projected' 2D magnetisation (nxn)
    s = Quiver plot skip density
    size = Arrow length scaling
    width = Arrow thickness 
    
    Output:
    2D Plot of magnetisation:
    - Arrows show direction of M
    - Background color shows magnitude of M
    """
    # Project along z by averaging
    x_proj = np.mean(X,axis=2)
    y_proj = np.mean(Y,axis=2)
    z_proj = np.mean(Z,axis=2)
    p_proj = np.mean(P,axis=2)
    
    if ax == None:
        # Create figure
        fig,ax = plt.subplots(figsize=(6, 8))

    # Plot magnitude
    im1 = ax.imshow(np.flipud(p_proj.T),vmin=0,vmax=1,cmap='Blues',
                     extent=(np.min(x_proj),np.max(x_proj),np.min(y_proj),np.max(y_proj)))
    
    # Add colorbar and labels
    clb = fig.colorbar(im1,ax=ax,fraction=0.046, pad=0.04)
    ax.set_xlabel('x / nm',fontsize=14)
    ax.set_ylabel('y / nm',fontsize=14)
    ax.set_title(title, fontsize= 16)

#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
    
    plt.tight_layout()
    
def rotate_bulk(P,ax,ay,az):
    """ 
    Rotate magnetisation locations from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    
    NOTE: This implementation of scipy rotations is EXTRINSIC
    Therefore, to make it compatible with our intrinsic vector
    rotation, we swap the order of rotations (i.e. x then y then z)
    """
    # Due to indexing, ay needs reversing for desired behaviour
    ay = -ay
    
    P = ndimage.rotate(P,ax,reshape=False,axes=(1,2),order=1)
    P = ndimage.rotate(P,ay,reshape=False,axes=(2,0),order=1)
    P = ndimage.rotate(P,az,reshape=False,axes=(0,1),order=1)

    return P

def plot_plane(P,ax,v=[0,0,1]):
    x,y,z = v
    y = -y
    s=5
    # create x,y
    xx, yy = np.meshgrid(np.linspace(15/s,85/s,5), np.linspace(15/s,85/s,5))

    normal = [x,y,z]
    d = -np.array([50/s,50/s,50/s]).dot(normal)

    # calculate corresponding z
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

    ax.plot_surface(xx, yy, zz, alpha=0.2,color='salmon')


    ax.plot([50/s,(50+50*x)/s],[50/s,(50+50*y)/s],[50/s,(50+50*z)/s],color='k')
    ax.plot([(50+50*x)/s],[(50+50*y)/s],[(50+50*z)/s],'o',color='red')


    im = ax.voxels(P[::s,::s,::s], facecolors=[0,0,1,.1], edgecolor=[1,1,1,0.1])



    # Add axis labels
    plt.xlabel('x / nm', fontsize=15)
    plt.ylabel('y / nm', fontsize=15)
    ax.set_zlabel('z / nm', fontsize=15)

    ax.set_xlim([0,100/s])
    ax.set_ylim([0,100/s])
    ax.set_zlim([0,100/s])

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    
def plot_both(X,Y,Z,P,ax,ay,az,save_path=None):
    # plot in 3D for a single tilt
    fig= plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    vx,vy,vz = angle_to_vector(ax,ay,az)
    Prot=rotate_bulk(P,ax,ay,az)
    plot_2d(X,Y,Z,Prot,ax=ax1,fig=fig)
    plot_plane(Prot,ax2,v=[vx,vy,vz])

    title = 'Projected density $(%i^{\circ},%i^{\circ},%i^{\circ})$' % (ax,ay,az)
    ax1.set_title(title,size=14)
    ax2.set_title('Projection direction visualised',size=14)
    
    if save_path != None:
        plt.savefig(save_path,bbox_inches='tight')
        
def angle_to_vector(ax,ay,az):
    θ = az * np.pi/180 # yaw
    ϕ = ax * np.pi/180 # pitch
    ψ = ay * np.pi/180 # roll
    
    x = -np.sin(ψ)*np.cos(θ)-np.cos(ψ)*np.sin(ϕ)*np.sin(θ)
    y = np.sin(ψ)*np.sin(θ)-np.cos(ψ)*np.sin(ϕ)*np.cos(θ)
    z = np.cos(ψ)*np.cos(ϕ)
    
    return x,y,z

def rotation_matrix(ax,ay,az,intrinsic=True):
    """ 
    Generate 3D rotation matrix from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    (Uses convention of rotating about z, then y, then x)
    """

    ax = ax * np.pi/180
    Cx = np.cos(ax)
    Sx = np.sin(ax)
    mrotx = np.array([[1,0,0],[0,Cx,-Sx],[0,Sx,Cx]])
    
    ay = ay * np.pi/180
    Cy = np.cos(ay)
    Sy = np.sin(ay)
    mroty = np.array([[Cy,0,Sy],[0,1,0],[-Sy,0,Cy]])
    
    az = az * np.pi/180
    Cz = np.cos(az)
    Sz = np.sin(az)
    mrotz = np.array([[Cz,-Sz,0],[Sz,Cz,0],[0,0,1]])
    
    if intrinsic == True:
        mrot = mrotz.dot(mroty).dot(mrotx)
    else:
        # To define mrot in an extrinsic space, matching
        # our desire for intrinsic rotation, we need
        # to swap the order of the applied rotations
        mrot = mrotx.dot(mroty).dot(mrotz)
    
    return mrot

def get_astravec(ax,ay,az):
    """ Given angles in degrees, return r,d,u,v as a concatenation
    of four 3-component vectors"""
    # Since we us flipud on y axis, ay needs reversing for desired behaviour
    ay = -ay 
    
    # centre of detector
    d = [0,0,0]
    
    # 3D rotation matrix - EXTRINSIC!
    mrot = np.array(rotation_matrix(ax,ay,az,intrinsic=False))
    
    # ray direction r
    r = mrot.dot([0,0,1])*-1 # -1 to match astra definitions
    # u (det +x)
    u = mrot.dot([1,0,0])
    # v (det +y)
    v = mrot.dot([0,1,0])

    return np.concatenate((r,d,u,v))

def generate_angles(mode='x',n_tilt = 40, alpha=70,beta=40,gamma=180,dist_n2=8,tilt2='gamma'):
    """ Return a list of [ax,ay,az] lists, each corresponding to axial
    rotations applied to [0,0,1] to get a new projection direction.
    
    Modes = x, y, dual, quad, sync, dist, rand
    
    Specify the +- tilt range of alpha/beta/gamma
    
    Say total number of tilts n_tilt
    
    For dist, each alpha has 'dist_n2' 'tilt2' projections
    
    Specify if the 2nd tilt axis is beta or gamma """
    
    angles = []
    ax,ay,az = 0,0,0
    
    # x series
    if mode=='x':
        for ax in np.linspace(-alpha,alpha,n_tilt):
            angles.append([ax,ay,az])
            
    if mode=='y':
        if tilt2 == 'beta':
            for ay in np.linspace(-beta,beta,n_tilt):
                angles.append([ax,ay,az])
        if tilt2 == 'gamma':
            if gamma >= 90:
                az = 90
            else:
                az = gamma
            for ax in np.linspace(-alpha,alpha,n_tilt):
                angles.append([ax,ay,az])
            
    if mode=='dual':
        for ax in np.linspace(-alpha,alpha,n_tilt/2):
            angles.append([ax,ay,az])
            
        ax,ay,az = 0,0,0
        if tilt2 == 'beta':
            for ay in np.linspace(-beta,beta,n_tilt/2):
                angles.append([ax,ay,az])
        if tilt2 == 'gamma':
            if gamma >=90:
                az = 90
            else:
                az = gamma
            for ax in np.linspace(-alpha,alpha,n_tilt/2):
                angles.append([ax,ay,az])
    
    if mode=='quad':
        if tilt2 == 'beta':
            for ax in np.linspace(-alpha,alpha,n_tilt/4):
                angles.append([ax,ay,az])
            ax,ay,az = 0,0,0
            for ay in np.linspace(-beta,beta,n_tilt/4):
                angles.append([ax,ay,az])
            ay = beta
            for ax in np.linspace(-alpha,alpha,n_tilt/4):
                angles.append([ax,ay,az])
            ay = -beta
            for ax in np.linspace(-alpha,alpha,n_tilt/4):
                angles.append([ax,ay,az])
                    
        if tilt2 == 'gamma':
            if gamma >= 90:
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = 90
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = 45
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = -45
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                        angles.append([ax,ay,az])           
            else:
                az = gamma
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = -gamma
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = gamma/3
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = -gamma/3
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])

    # random series # g or b
    if mode=='rand':
        for i in range(n_tilt):
            ax_rand = np.random.rand()*alpha*2 - alpha
            if tilt2 == 'beta':
                ay_rand = np.random.rand()*beta*2 - beta
                angles.append([ax_rand,ay_rand,0])
            if tilt2 == 'gamma':
                az_rand = np.random.rand()*gamma*2 - gamma
                angles.append([ax_rand,0,az_rand])
            
    # alpha propto beta series # g or b
    if mode=='sync':
        if tilt2 == 'beta': 
            ax = np.linspace(-alpha,alpha,n_tilt/2)
            ay = np.linspace(-beta,beta,n_tilt/2)

            for i,a in enumerate(ax):
                angles.append([a,ay[i],0])

            for i,a in enumerate(ax):
                angles.append([a,-ay[i],0])
        if tilt2 == 'gamma': 
            ax = np.linspace(-alpha,alpha,n_tilt/2)
            az = np.linspace(-gamma,gamma,n_tilt/2)

            for i,a in enumerate(ax):
                angles.append([a,0,az[i]])

            for i,a in enumerate(ax):
                angles.append([a,0,-az[i]])
            
    # even spacing # g or b
    if mode=='dist':
        ax = np.linspace(-alpha,alpha,n_tilt/dist_n2)
        if alpha == 90:
            ax = np.linspace(-90,90,n_tilt/dist_n2+1)
            ax = ax[::-1]
        if tilt2 == 'beta': 
            ay = np.linspace(-beta,beta,dist_n2)
            for x in ax:
                for y in ay:
                    angles.append([x,y,0])
        if tilt2 == 'gamma': 
            if gamma < 90:
                az = np.linspace(-gamma,gamma,dist_n2)
                for x in ax:
                    for z in az:
                        angles.append([x,0,z])
            if gamma >= 90:
                az = np.linspace(-90,90,dist_n2+1)
                for x in ax:
                    for z in az[:-1]:
                        angles.append([x,0,z])
    
    return angles

def generate_proj_data(P,angles):
    """ Returns projection dataset given phantom P
    and 3D projection angles list.
    
    Output is normalised and reshaped such that the
    projection slice dimension is in the middle, so as
    to be compatible with astra."""
    P_projs = []
    
    for i in range(len(angles)):
        ax,ay,az = angles[i]
        P_rot = rotate_bulk(P,ax,ay,az) 
        P_rot_proj =np.flipud(np.mean(P_rot,axis=2).T) #flip/T match data shape to expectations
        P_projs.append(P_rot_proj) 
        
    # Prepare projections for reconstruction
    raw_data = np.array(P_projs)
    raw_data = raw_data -  raw_data.min()
    raw_data = raw_data/raw_data.max()
    raw_data = np.transpose(raw_data,axes=[1,0,2]) # reshape so proj is middle column
        
    return raw_data
      
def generate_vectors(angles):
    """ Converts list of 3D projection angles into
    list of astra-compatible projection vectors,
    with [r,d,u,v] vectors on each row. """
    vectors = []
    for [ax,ay,az] in angles:
        vector = get_astravec(ax,ay,az)
        vectors.append(vector)
    
    return vectors

def generate_reconstruction(raw_data,vectors, algorithm = 'SIRT3D_CUDA', niter=10, weight = 0.01,
                            balance = 1, steps = 'backtrack', callback_freq = 0):
    """ Chooise from 'SIRT3D_CUDA','FP3D_CUDA','BP3D_CUDA','CGLS3D_CUDA' or 'TV1'"""
    # Astra default algorithms
    if algorithm in ['SIRT3D_CUDA','FP3D_CUDA','BP3D_CUDA','CGLS3D_CUDA']:
        # Load data objects into astra C layer
        proj_geom = astra.create_proj_geom('parallel3d_vec',np.shape(raw_data)[0],np.shape(raw_data)[2],np.array(vectors))
        projections_id = astra.data3d.create('-sino', proj_geom, raw_data)
        vol_geom = astra.creators.create_vol_geom(np.shape(raw_data)[0], np.shape(raw_data)[0],
                                                  np.shape(raw_data)[2])
        reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
        alg_cfg = astra.astra_dict(algorithm)
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)

        astra.algorithm.run(algorithm_id,iterations=niter)
        recon = astra.data3d.get(reconstruction_id)
    
    # CS TV using RTR
    if algorithm == 'TV1':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV(vol_shape, order=1)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
    
    if algorithm == 'TV2':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV(vol_shape, order=2)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
    
    return recon

def reorient_reconstruction(r):
    # Swap columns back to match orientation of phantom
    r = np.transpose(r,[2,1,0]) # Reverse column order
    r = r[:,::-1,:] # Reverse the y data
    r = r -  r.min() # normalise
    r = r/r.max()

    recon_vector = copy.deepcopy(r)
    
    return recon_vector

def COD(P,recon):
    """ Calculate the coefficinet of determination (1 perfect, 0 shit)"""
    P_mean = np.mean(P)
    R_mean = np.mean(recon)
    sumprod = np.sum((P-P_mean)*(recon-R_mean))
    geom_mean = np.sqrt(np.sum((P-P_mean)**2)*np.sum((recon-R_mean)**2))
    coeff_norm = sumprod/geom_mean
    COD = coeff_norm**2
    
    return COD

def error_opt(beta,recon,P):
    a = np.linalg.norm(recon*beta-P)
    b = np.linalg.norm(P)
    return a/b

def phantom_error(P,recon,beta=1):
    """ Calculate normalised error between phantom and reconstruction
    (0 great, 1 shit) """
    opt = optimize.minimize(error_opt,1,args=(recon,P))
    err_phant = opt.fun
    return err_phant

def projection_error(P,recon,angles,beta=1):
    """ Calculate normalised error between phantom projections and reconstruction
    projections (0 great, 1 shit) """
    true_proj = generate_proj_data(P,angles)
    recon_proj = generate_proj_data(recon,angles)
    err_proj = phantom_error(true_proj,recon_proj,1)
    return err_proj

def noisy(image, noise_typ='gauss',g_var = 0.1, p_sp = 0.004,val_pois = None,sp_var=1):
    """ Add noise to image with choice from:
    - 'gauss' for Gaussian noise w/ variance 'g_var'
    - 's&p' for salt & pepper noise with probability 'p_sp'
    - 'poisson' for shot noise with avg count of 'val_pois'
    - 'speckle' for speckle noise w/ variance 'sp_var'"""
    if noise_typ == "gauss":
        # INDEPENDENT (ADDITIVE)
        # Draw random samples from a Gaussian distribution
        # Add these to the image
        # Higher variance = more noise
        row,col,ch= image.shape
        mean = 0
        var = g_var
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    
    elif noise_typ == "s&p":
        # INDEPENDENT
        # Salt & pepper/spike/dropout noise will either
        # set random pixels to their max (salt) or min (pepper)
        # Quantified by the % of corrupted pixels
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = p_sp
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape] # randomly select coordinates
        out[coords] = np.max(image) # set value to max

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape] # randomly select coordinates
        out[coords] = np.min(image) # set value to min
        return out
    
    elif noise_typ == "poisson":
        # DEPENDENT (MULTIPLICATIVE)
        # Poisson noise or shot noise arises due to the quantised
        # nature of particle detection.
        # Each pixel changes from its original value to 
        # a value taken from a Poisson distrubution with
        # the same mean (multiplied by vals)
        # So val can be thought of as the avg no. of electrons
        # contributing to that pixel of the image (low = noisy)
        if val_pois == None:
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
        else:
            vals = val_pois
            
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    
    elif noise_typ =="speckle":
        # DEPENDENT (MULTIPLICATIVE)
        # Random value multiplications of the image pixels
        
        # Generate array in shape of image but with values
        # drawn from a Gaussian distribution
        row,col,ch= image.shape
        mean = 0
        var = sp_var
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        
        # Multiply image by dist. and add to image
        noisy = image + image * gauss
        return noisy
    
def vec_to_ang(v):
    """ Returns a set of Euler rotations that map [0,0,1] to V
    Note: This will not be unique, but will be 'an' answer.
    
    https://stackoverflow.com/questions/51565760/euler-angles-and-rotation-matrix-from-two-3d-points 
    It works by first finding the axis of rotation from AxB,
    then getting the angle with atan(AxB/A.B).
    It then converts this to a rotation matrix and finally to 
    Euler angles using another module"""
    A = np.array([0,0,1])
    B = np.array(v)

    cross = np.cross(A, B)
    dot = np.dot(A, B.transpose())
    angle = np.arctan2(np.linalg.norm(cross), dot)
    rotation_axes = normalize(cross)
    rotation_m = transforms3d.axangles.axangle2mat(rotation_axes, angle, True)
    rotation_angles = transforms3d.euler.mat2euler(rotation_m, 'sxyz')
    
    return np.array(rotation_angles)*180/np.pi

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def compare_projection(recon_vector,P,ax=0,ay=0,az=0):
    """ Plots reconstruction and phantom side by side and prints error metrics """
    a = rotate_bulk(recon_vector,ax,ay,az)

    fig= plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(np.flipud(np.mean(a,axis=2).T))
    ax2.imshow(np.flipud(np.mean(rotate_bulk(P,ax,ay,az),axis=2).T))
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    ax1.set_title('Reconstruction',fontsize=14)
    ax2.set_title('Phantom',fontsize=14)

    print('Phantom error: ',phantom_error(P,recon_vector),'COD: ',COD(P,recon_vector))
    
def compare_ortho(P,r,ax=0,ay=0,az=0,ix=None,iy=None,iz=None):
    """ Plot recon orthoslices above phantom orthoslices and print error metrics"""
    
    Prot = rotate_bulk(P,ax,ay,az)
    rrot = rotate_bulk(r,ax,ay,az)
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3)
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    ax6 = fig.add_subplot(2,3,6)

    plot_orthoslices(rrot,axs=[ax1,ax2,ax3],ix=ix,iy=iy,iz=iz)
    plot_orthoslices(Prot,axs=[ax4,ax5,ax6],ix=ix,iy=iy,iz=iz)
    
    ax3.set_title('YZ - Recon',fontsize=15,weight='bold')
    ax2.set_title('XZ - Recon',fontsize=15,weight='bold')
    ax1.set_title('XY - Recon',fontsize=15,weight='bold')
    
    ax6.set_title('YZ - Phantom',fontsize=15,weight='bold')
    ax5.set_title('XZ - Phantom',fontsize=15,weight='bold')
    ax4.set_title('XY - Phantom',fontsize=15,weight='bold')
    
    plt.tight_layout()
    print('Phantom error: ',phantom_error(P,r),'COD: ',COD(P,r))
    
def plot_orthoslices(P,ix=None,iy=None,iz=None,axs=None):
    """ Plot xy,xz,yz orthoslices of a 3d volume
    Plots central slice by default, but slice can be specified """
    if axs == None:
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
    else:
        ax1,ax2,ax3 = axs
        fig = plt.gcf()
        
    pmax, pmin = np.max(P),np.min(P)

    sx,sy,sz = np.shape(P)
    sx2 = int(sx/2)
    sy2 = int(sy/2)
    sz2 = int(sz/2)
    
    if ix != None:
        sx2 = ix
    if iy != None:
        sy2 = iy
    if iz != None:
        sz2 = iz

    ax3.imshow(P[sx2,:,:],cmap='Greys_r',vmax=pmax,vmin=pmin)
    ax2.imshow(P[:,sy2,:],cmap='Greys_r',vmax=pmax,vmin=pmin)
    ax1.imshow(P[:,:,sz2],cmap='Greys_r',vmax=pmax,vmin=pmin)
    
    

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.tight_layout()

    ax3.set_title('YZ',fontsize=15,weight='bold')
    ax2.set_title('XZ',fontsize=15,weight='bold')
    ax1.set_title('XY',fontsize=15,weight='bold')
    
    fig.patch.set_facecolor('0.9')
    
def full_tomo(P,Pn,scheme='x',a=70,b=40,g=180,alg='TV1',tilt2='gamma',n_tilt=40,angles = None,dist_n2=8,niter=300,callback_freq=50,weight=0.01):
    """ Given a phantom, returns reconstructed volume (and projection data and angles)"""
    if angles == None:
        angles = generate_angles(mode=scheme,alpha=a,beta=b,gamma=g,tilt2=tilt2,dist_n2=dist_n2,n_tilt=n_tilt)
    raw_data = generate_proj_data(Pn,angles)
    vectors = generate_vectors(angles)
    recon = generate_reconstruction(raw_data,vectors,algorithm=alg,niter=niter,callback_freq=callback_freq,weight=weight)
    recon_vector = reorient_reconstruction(recon)
    return [recon_vector,raw_data,angles]


### magnetic stuff starts

def convertmag_T_Am(M_T):
    """ Enter 'equivalent' magnetisation in Tesla
    for a conversion to A/m.
    Can also input common materials as string, e.g. 
    'cobalt' """
    if M_T == 'cobalt':
        M_Am = 1435860
        
    else:
        M_Am = M_T / (4*np.pi*1e-7)
        
    return M_Am



class Magnetic_Phantom():
    """ Class for creating magnetic phantoms """

    def sphere(rad_m = 10*1e-9, Ms_Am = 797700, plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates uniformly magnetised sphere
            rad_m : Radius in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 + (k-ci)**2 < (rad_m/res)**2:
                        MX[i,j,k] = np.cos(plan_rot*np.pi/180)*Ms_Am
                        MY[i,j,k] = np.sin(plan_rot*np.pi/180)*Ms_Am
        
        return MX,MY,MZ, mesh_params
    
    def rectangle(lx_m = 80*1e-9,ly_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, p2 = (100*1e-9,100*1e-9,100*1e-9), n=(100,100,100)):
        """ Creates uniformly magnetised rectangle
            l_m = length of rectangle in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        # Initialise bounding box parameters
        p1 = (0,0,0)
        mesh_params = [p1,p2,n]
        resx = p2[0]/n[0] # resolution in m per px 
        resy = p2[1]/n[1] # resolution in m per px 
        resz = p2[2]/n[2] # resolution in m per px 
        cix = int(n[0]/2) # index of bbox centre
        ciy = int(n[1]/2) # index of bbox centre
        ciz = int(n[2]/2) # index of bbox centre

        # Initialise magnetisation arrays
        mx = np.linspace(0,p2[0],num=n[0]) * 0
        my = np.linspace(0,p2[1],num=n[1]) * 0
        mz = np.linspace(0,p2[2],num=n[2]) * 0
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if cix-.5*lx_m/resx < i < cix+.5*lx_m/resx and \
                       ciy-.5*ly_m/resy < j < ciy+.5*ly_m/resy and \
                       ciz-.5*lz_m/resz < k < ciz+.5*lz_m/resz:
                        MX[i,j,k] = np.cos(plan_rot*np.pi/180)*Ms_Am
                        MY[i,j,k] = np.sin(plan_rot*np.pi/180)*Ms_Am

        return MX,MY,MZ, mesh_params
    
    def disc_vortex(rad_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates disk with c/w vortex magnetisation
            rad_m : Radius in metres
            lz_m = thickness of disc in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        def vortex(x,y):
            """ Returns mx/my components for vortex state, 
            given input x and y """
            # angle between tangent and horizontal
            theta=-1*np.arctan2(x,y)
            # cosine/sine components
            C=np.cos(theta)
            S = np.sin(theta)
            return C, S
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 < (rad_m/res)**2 and ci-.5*lz_m/res < k < ci+.5*lz_m/res:
                        mx,my = vortex(i-ci,j-ci)
                        MX[i,j,k] = mx*Ms_Am
                        MY[i,j,k] = my*Ms_Am
                        
        
        return MX,MY,MZ, mesh_params
    
    def disc_uniform(rad_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates disk with c/w vortex magnetisation
            rad_m : Radius in metres
            lz_m = thickness of disc in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 < (rad_m/res)**2 and ci-.5*lz_m/res < k < ci+.5*lz_m/res:
                        MX[i,j,k] = np.cos(plan_rot*np.pi/180)*Ms_Am
                        MY[i,j,k] = np.sin(plan_rot*np.pi/180)*Ms_Am
                        
        
        return MX,MY,MZ, mesh_params
    
    def tri_pris(rad_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates disk with c/w vortex magnetisation
            rad_m : Radius in metres
            lz_m = thickness of disc in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')
        
        # Define gradient/intercept of bounding lines
        m1, c1 = 5/(100*1e-9)*bbox_length_m,   100 /100*bbox_length_px
        m2, c2 = 0,                            -25 /100*bbox_length_px
        m3, c3 = -0.6/(100*1e-9)*bbox_length_m, 0

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    x = i-ci
                    y = j-ci
                    z = k-ci
                    if y < (m1*x+c1) and y > (m2*x + c2) and y < (m3*x + c3) and ((z >-20/100*bbox_length_px and z<-10/100*bbox_length_px) or (z>0 and z<30/100*bbox_length_px)):
                        MX[i,j,k] = Ms_Am
                        
        #MX = np.swapaxes(MX,0,1)
                        
        return MX,MY,MZ, mesh_params
    
    def rod(rad_m = 10*1e-9, lx_m = 60*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates uniformly magnetised cylindrical rod lying along x
            rad_m : Radius in metres
            lx_m = length of rod in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    x = i-ci
                    y = j-ci
                    z = k-ci
                    if (k-ci)**2 + (j-ci)**2 < (rad_m/res)**2 and ci-.5*lx_m/res < i < ci+.5*lx_m/res:
                        MX[i,j,k] = Ms_Am
                        
        #MX = np.swapaxes(MX,0,1)
                        
        return MX,MY,MZ, mesh_params

def plot_2d_mag(mx,my,mesh_params=None,Ms=None,s=1):
    """ Takes x/y magnetisation projections and creates a plot
        uses quivers for direction and colour for magnitude """
    if type(Ms) == type(None):
        Ms = np.max(np.max((mx**2+my**2)**.5))
    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy = np.shape(mx)
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
        
    x = np.linspace(p1[0],p2[0],num=n[0])
    y = np.linspace(p1[1],p2[1],num=n[1])
    xs,ys = np.meshgrid(x,y)
    
    plt.quiver(xs[::s,::s],ys[::s,::s],mx[::s,::s].T,my[::s,::s].T,pivot='mid',scale=Ms*22,width=0.009,headaxislength=5,headwidth=4,minshaft=1.8)
    mag = (mx**2+my**2)**.5
    plt.imshow(mag.T,origin='lower',extent=[p1[0],p2[0],p1[1],p2[1]],vmin=0,vmax=Ms,cmap='Blues')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('$|M_{\perp}$| / $A $', rotation=-270,fontsize=15)
    
    plt.xlabel('x / m',fontsize=15)
    plt.ylabel('y / m',fontsize=15)
    plt.show()
    
def project_along_z(U,mesh_params=None):
    """ Takes a 3D array and projects along the z component 
    It does this by multiplying each layer by its thickness
    and then summing down the axis. """
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy,sz = np.shape(U)
        p2 = (sx,sy,sz)
        n = p2
    else:
        p1,p2,n = mesh_params
    
    # Get resolution    
    z_size = p2[2]
    z_res = z_size/n[2]
    
    # project
    u_proj = np.sum(U*z_res,axis=2)
    
    return u_proj

def calculate_A_3D(MX,MY,MZ, mesh_params=None,n_pad=100,tik_filter=0.01):
    """ Input(3D (nx,ny,nz) array for each component of M) and return
    three 3D arrays of magnetic vector potential 
    
    Note, returned arrays will remain padded since if they are used for
    projection to phase change this will make a difference. So the new
    mesh parameters are also returned
    
    """
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy,sz = np.shape(MX)
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
    
    # zero pad M to avoid FT convolution wrap-around artefacts
    mxpad = np.pad(MX,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    mypad = np.pad(MY,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    mzpad = np.pad(MZ,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)

    # take 3D FT of M    
    ft_mx = np.fft.fftn(mxpad)
    ft_my = np.fft.fftn(mypad)
    ft_mz = np.fft.fftn(mzpad)
    
    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 

    kx = np.fft.fftfreq(ft_mx.shape[0],d=resx)
    ky = np.fft.fftfreq(ft_my.shape[0],d=resy)
    kz = np.fft.fftfreq(ft_mz.shape[0],d=resz)
    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij') # Create a grid of coordinates
    
    # vacuum permeability
    mu0 = 4*np.pi*1e-7
    
    # Calculate 1/k^2 with Tikhanov filter
    if tik_filter == 0:
        K2_inv = np.nan_to_num(((KX**2+KY**2+KZ**2)**.5)**-2)
    else:
        K2_inv = ((KX**2+KY**2+KZ**2)**.5 + tik_filter*resx)**-2
    
    # M cross K
    cross_x = ft_my*KZ - ft_mz*KY
    cross_y = -ft_mx*KZ + ft_mz*KX
    cross_z = -ft_my*KX + ft_mx*KY
    
    # Calculate A(k)
    ft_Ax = (-1j * mu0 * K2_inv) * cross_x
    ft_Ay = (-1j * mu0 * K2_inv) * cross_y
    ft_Az = (-1j * mu0 * K2_inv) * cross_z
    
    # Inverse fourier transform
    Ax = np.fft.ifftn(ft_Ax)
    AX = Ax.real
    Ay = np.fft.ifftn(ft_Ay)
    AY = Ay.real
    Az = np.fft.ifftn(ft_Az)
    AZ = Az.real
    
    # new mesh parameters (with padding)
    n = (n[0]+2*n_pad,n[1]+2*n_pad,n[2]+2*n_pad)
    p2 = (p2[0]+2*n_pad*resx,p2[1]+2*n_pad*resy,p2[2]+2*n_pad*resz)
    mesh_params=(p1,p2,n)
    
    return AX,AY,AZ,mesh_params

def calculate_phase_AZ(AZ,mesh_params=None):
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy,sz = np.shape(MX)
        p2 = (sx,sy,sx)
        n = p2
        mesh_params = [p1,p2,n]
    else:
        p1,p2,n = mesh_params
    """ Calculates projected phase change from 3D AZ """
    AZ_proj = project_along_z(AZ,mesh_params=mesh_params) 
    phase = AZ_proj * -1* np.pi/constants.codata.value('mag. flux quantum') / (2*np.pi)
    return phase

def calculate_phase_M_2D(MX,MY,MZ,mesh_params,n_pad=500,tik_filter=0.01,unpad=True):
    """ Preffered method. Takes 3D MX,MY,MZ magnetisation arrays
    and calculates phase shift in rads in z direction.
    First projects M from 3D to 2D which speeds up calculations """
    p1,p2,n=mesh_params
    
    # J. Loudon et al, magnetic imaging, eq. 29
    const = .5 * 1j * 4*np.pi*1e-7 / constants.codata.value('mag. flux quantum')

    # Define resolution from mesh parameters
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    
    # Project magnetisation array
    mx = project_along_z(MX,mesh_params=mesh_params)
    my = project_along_z(MY,mesh_params=mesh_params)
    
    # Take fourier transform of M
    # Padding necessary to stop Fourier convolution wraparound (spectral leakage)
    if n_pad > 0:
        mx = np.pad(mx,[(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
        my = np.pad(my,[(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    
    ft_mx = np.fft.fft2(mx)
    ft_my = np.fft.fft2(my)
    
    # Generate K values
    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)
    KX, KY = np.meshgrid(kx,ky, indexing='ij') # Create a grid of coordinates
    
    # Filter to avoid division by 0
    if tik_filter == 0:
        K2_inv = np.nan_to_num(((KX**2+KY**2)**.5)**-2)
    else:
        K2_inv = ((KX**2+KY**2)**.5 + tik_filter*resx)**-2

    # Take cross product (we only need z component)
    cross_z = (-ft_my*KX + ft_mx*KY)*K2_inv
    
    # Inverse fourier transform
    phase = np.fft.ifft2(const*cross_z).real
    
    # Unpad
    if unpad == True:
        if n_pad > 0:
            phase=phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def calculate_phase_M_3D(MX,MY,MZ,mesh_params,n_pad=100,tik_filter=0.01):
    """ Slower than 2D but good for comparison. Takes 3D MX,MY,MZ magnetisation arrays
    and calculates phase shift in rads in z direction.
    Calculations performed directly in 3D """
    p1,p2,n=mesh_params
    
    # constant prefactor
    const = .5*1j*4*np.pi*1e-7/constants.codata.value('mag. flux quantum')

    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    MX = np.pad(MX,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    MY = np.pad(MY,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    MZ = np.pad(MZ, [(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)
    kz = np.fft.fftfreq(n[2]+2*n_pad,d=resz)
    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij') # Create a grid of coordinates
    K2_inv = np.nan_to_num(((KX**2+KY**2+KZ**2)**.5+ tik_filter*resx)**-2)
    
    # Take 3D fourier transforms (only need x and y for cross-z)
    ft_mx = np.fft.fftn(MX)
    ft_my = np.fft.fftn(MY)
    
    # take cross product
    cross_z = (-ft_my*KX + ft_mx*KY) * K2_inv
    
    # extract central slice
    slice_z = cross_z[:,:,0] * resz 
    
    # inverse fourier transform
    phase = np.fft.ifft2(const*slice_z).real
    
    if n_pad > 0:
        phase = phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def analytical_sphere(B0_T=1.6,r_m=50*1e-9,mesh_params=None,beta=-90,n_pad=100):
    """ Analytically calculates the phase change for a sphere (from Beleggia and Zhu 2003)"""
    import scipy
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(AX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    p1,p2,n=mesh_params
    
    # Calculate prefactor
    const = 4 * np.pi**2 * 1j * B0_T  *(r_m/2/np.pi)**2/ constants.codata.value('mag. flux quantum')
    
    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    
    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)#/(2*np.pi))
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)#/(2*np.pi))
    KX, KY = np.meshgrid(kx,ky)#,indexing='ij') # Create a grid of coordinates
    
    # Calculate 1/k^2 with Tikhanov filter
    K3_inv = np.nan_to_num(((KX**2+KY**2)**.5)**-3)
    K =(KX**2+KY**2)**.5

    #The normalized sinc function is the Fourier transform of the rectangular function with no scaling. np default is normalised
    phase_ft = const * (KY*np.cos(beta*np.pi/180) - KX*np.sin(beta*np.pi/180)) * K3_inv \
                        * scipy.special.spherical_jn(1,r_m*(2*np.pi)*K) / (resx*resy)
    
    phase = np.fft.ifft2(phase_ft).real 
    
    phase = np.fft.ifftshift(phase) 
    
    phase = phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def analytical_rectangle(B0_T=1.6,lx_m=200*1e-9,ly_m=140*1e-9,lz_m=20*1e-9,mesh_params=None,beta=300,n_pad=100):
    """ Analytically calculates the phase change for a rectangle (from Beleggia and Zhu 2003)"""
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(AX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    p1,p2,n=mesh_params
    
    # Calculate prefactor
    V = lx_m*ly_m*lz_m
    const = 1j*np.pi*B0_T*V/constants.codata.value('mag. flux quantum')
    
    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 

    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)
    KX, KY = np.meshgrid(kx,ky,indexing='ij') # Create a grid of coordinates
    #KX,KY=KX*(2*np.pi),KY*(2*np.pi)
    
    # Calculate 1/k^2 with Tikhanov filter
    K2_inv = np.nan_to_num(((KX**2+KY**2)**.5)**-2)

    #The normalized sinc function is the Fourier transform of the rectangular function with no scaling. np default is normalised
    phase_ft = const * K2_inv * (KY*np.cos(beta*np.pi/180) - KX*np.sin(beta*np.pi/180)) * np.sinc(lx_m*KX) * np.sinc(ly_m*KY) / (resx*resy)
    
    phase = np.fft.ifft2(phase_ft).real
    
    phase = np.fft.ifftshift(phase) / (2*np.pi)
    
    if n_pad>0:
        phase = phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def linsupPhi(mx=1.0, my=1.0, mz=1.0, Dshp=None, theta_x=0.0, theta_y=0.0, pre_B=1.0, pre_E=1, v=1, multiproc=True):
    """Applies linear superposition principle for 3D reconstruction of magnetic and electrostatic phase shifts.
    This function will take 3D arrays with Mx, My and Mz components of the 
    magnetization, the Dshp array consisting of the shape function for the 
    object (1 inside, 0 outside), and the tilt angles about x and y axes to 
    compute the magnetic and the electrostatic phase shift. Initial computation 
    is done in Fourier space and then real space values are returned.
    Args: 
        mx (3D array): x component of magnetization at each voxel (z,y,x)
        my (3D array): y component of magnetization at each voxel (z,y,x)
        mz (3D array): z component of magnetization at each voxel (z,y,x)
        Dshp (3D array): Binary shape function of the object. Where value is 0,
            phase is not computed.  
        theta_x (float): Rotation around x-axis (degrees). Rotates around x axis
            then y axis if both are nonzero. 
        theta_y (float): Rotation around y-axis (degrees) 
        pre_B (float): Numerical prefactor for unit conversion in calculating 
            the magnetic phase shift. Units 1/pixels^2. Generally 
            (2*pi*b0*(nm/pix)^2)/phi0 , where b0 is the Saturation induction and 
            phi0 the magnetic flux quantum. 
        pre_E (float): Numerical prefactor for unit conversion in calculating the 
            electrostatic phase shift. Equal to sigma*V0, where sigma is the 
            interaction constant of the given TEM accelerating voltage (an 
            attribute of the microscope class), and V0 the mean inner potential.
        v (int): Verbosity. v >= 1 will print status and progress when running
            without numba. v=0 will suppress all prints. 
        mp (bool): Whether or not to implement multiprocessing. 
    Returns: 
        tuple: Tuple of length 2: (ephi, mphi). Where ephi and mphi are 2D numpy
        arrays of the electrostatic and magnetic phase shifts respectively. 
    """
    import time
    vprint = print if v>=1 else lambda *a, **k: None
    [dimz,dimy,dimx] = mx.shape
    dx2 = dimx//2
    dy2 = dimy//2
    dz2 = dimz//2

    ly = (np.arange(dimy)-dy2)/dimy
    lx = (np.arange(dimx)-dx2)/dimx
    [Y,X] = np.meshgrid(ly,lx, indexing='ij')
    dk = 2.0*np.pi # Kspace vector spacing
    KX = X*dk
    KY = Y*dk
    KK = np.sqrt(KX**2 + KY**2) # same as dist(ny, nx, shift=True)*2*np.pi
    zeros = np.where(KK == 0)   # but we need KX and KY later. 
    KK[zeros] = 1.0 # remove points where KK is zero as will divide by it

    # compute S arrays (will apply constants at very end)
    inv_KK =  1/KK**2
    Sx = 1j * KX * inv_KK
    Sy = 1j * KY * inv_KK
    Sx[zeros] = 0.0
    Sy[zeros] = 0.0
    
    # Get indices for which to calculate phase shift. Skip all pixels where
    # thickness == 0 
    if Dshp is None: 
        Dshp = np.ones(mx.shape)
    # exclude indices where thickness is 0, compile into list of ((z1,y1,x1), (z2,y2...
    zz, yy, xx = np.where(Dshp != 0)
    inds = np.dstack((zz,yy,xx)).squeeze()

    # Compute the rotation angles
    st = np.sin(np.deg2rad(theta_x))
    ct = np.cos(np.deg2rad(theta_x))
    sg = np.sin(np.deg2rad(theta_y))
    cg = np.cos(np.deg2rad(theta_y))

    x = np.arange(dimx) - dx2
    y = np.arange(dimy) - dy2
    z = np.arange(dimz) - dz2
    Z,Y,X = np.meshgrid(z,y,x, indexing='ij') # grid of actual positions (centered on 0)

    # compute the rotated values; 
    # here we apply rotation about X first, then about Y
    i_n = Z*sg*ct + Y*sg*st + X*cg
    j_n = Y*ct - Z*st

    mx_n = mx*cg + my*sg*st + mz*sg*ct
    my_n = my*ct - mz*st

    # setup 
    mphi_k = np.zeros(KK.shape,dtype=complex)
    ephi_k = np.zeros(KK.shape,dtype=complex)

    nelems = np.shape(inds)[0]
    stime = time.time()
    vprint(f'Beginning phase calculation for {nelems:g} voxels.')
    if multiproc:
        vprint("Running in parallel with numba.")
        ephi_k, mphi_k = exp_sum(mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx)        

    else:
        vprint("Running on 1 cpu.")
        otime = time.time()
        vprint('0.00%', end=' .. ')
        cc = -1
        for ind in inds:
            ind = tuple(ind)
            cc += 1
            if time.time() - otime >= 15:
                vprint(f'{cc/nelems*100:.2f}%', end=' .. ')
                otime = time.time()
            # compute the expontential summation
            sum_term = np.exp(-1j * (KY*j_n[ind] + KX*i_n[ind]))
            ephi_k += sum_term 
            mphi_k += sum_term * (my_n[ind]*Sx - mx_n[ind]*Sy)
        vprint('100.0%')

    vprint(f"total time: {time.time()-stime:.5g} sec, {(time.time()-stime)/nelems:.5g} sec/voxel.")
    #Now we have the phases in K-space. We convert to real space and return
    ephi_k[zeros] = 0.0
    mphi_k[zeros] = 0.0
    ephi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ephi_k)))).real*pre_E
    mphi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mphi_k)))).real*pre_B

    return (ephi,mphi)

def plot_phase_proj(phase,mesh_params=None,ax=None):
    """ Plots the projected phase shift in rads """
    if mesh_params == None:
            p1 = (0,0,0)
            sx,sy = np.shape(phase)
            p2 = (sx,sy,sx)
            n = p2
    else:
        p1,p2,n = mesh_params
        
    if ax == None:
        fig,ax = plt.subplots()
    fig=plt.gcf()

    im = ax.imshow(phase.T,extent=[p1[0],p2[0],p1[1],p2[1]],origin='lower')
    cbar = fig.colorbar(im,fraction=0.046, pad=0.04,ax=ax)

    cbar.set_label('Projected phase shift / rad', rotation=-270,fontsize=15)
    ax.set_xlabel('x / m',fontsize=14)
    ax.set_ylabel('y / m',fontsize=14)
    #plt.show()
    
def calculate_B_from_A(AX,AY,AZ,mesh_params=None):
    """ Takes curl of B to get A """
    # Initialise parameters
    phase_projs = []
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(AX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    p1,p2,n=mesh_params
    
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    
    BX = np.gradient(AZ,resy)[1] - np.gradient(AY,resz)[2]
    BY = np.gradient(AX,resz)[2] - np.gradient(AZ,resx)[0]
    BZ = np.gradient(AY,resx)[0] - np.gradient(AX,resy)[1]
        
    return BX/(2*np.pi),BY/(2*np.pi),BZ/(2*np.pi)

def calculate_B_from_phase(phase_B,mesh_params=None):
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy = np.shape(mx)
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
        
    x_size = p2[0]
    x_res = x_size/n[0]
    
    y_size = p2[1]
    y_res = y_size/n[1]
    
    d_phase = np.gradient(phase_B,x_res)
    b_const = (constants.codata.value('mag. flux quantum')/(np.pi))
    b_field_x = -b_const*d_phase[1]
    b_field_y = b_const*d_phase[0]

    mag_B = np.hypot(b_field_x,b_field_y)
    
    return mag_B,b_field_x,b_field_y

def plot_colorwheel(alpha=1,rot=0,flip= False, ax=None,rad=0.5,clip=48,shape=200,shift_centre=None,mesh_params=None):
    """ Plots a colorwheel
    alpha - match alpha to the alpha in your B plot (i.e. how black is the centre)
    rot - rotate the color wheel, enter angle in degrees (must be multiple of 90)
    flip - change between cw / ccw
    ax - pass an axis to plot on top of another image
    rad - radius of the colorwheel, scaled 0 to 1
    clip - radius of colorwheel clip in distance (m)
    shape - length of array size (must be square array)
    shift_centre - tuple lets you shift centre in px (horz,vert)
    """
    def cart2pol(x, y):
        """ Convert cartesian to polar coordinates
        rho = magnitude, phi = angle """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy =200,200
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
        
    extent = (p1[0],p2[0],p1[1],p2[1])
    
    ax_tog = 1
    if type(ax) == type(None):
        fig,ax = plt.subplots()
        ax_tog=0
        
    scale = (p2[0]-p1[0])
    centre = np.array((scale/2, scale/2)) #* scale/shape/100
    if type(shift_centre) == type(None):
        shift_centre=(0,0)
    shift_centre = np.array(shift_centre)/shape #* scale/shape/100
    
    # Create coordinate space
    x = np.linspace(-scale/2,scale/2,shape)
    y = x
    X,Y = np.meshgrid(x,y)

    # Map theta values onto coordinate space 
    thetas = np.ones_like(X)*0
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            # shifting will shift the centre of divergence
            thetas[ix,iy] = cart2pol((xx+shift_centre[0]*scale),(yy+shift_centre[1]*scale))[1]

    # Plot hsv colormap of angles
    if flip == False:
        im1 = ax.imshow(ndimage.rotate(thetas.T,180+rot),cmap='hsv_r',origin='lower',extent=extent,zorder=2)
    if flip == True:
        im1 = ax.imshow(ndimage.rotate(thetas,270+rot),cmap='hsv_r',origin='lower',extent=extent,zorder=2)

    # Create alpha contour map
    my_cmap = alpha_cmap()
    
    # Map circle radii onto xy coordinate space
    circ = np.ones_like(X)*0
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            if (xx+shift_centre[1]*scale)**2 + (yy-shift_centre[0]*scale)**2 < (rad*scale)**2:
                #print(xx,shift_centre[0])
                circ[ix,iy] = cart2pol((xx+shift_centre[1]*scale),(yy-shift_centre[0]*scale))[0]

    # Plot circle
    im2 = plt.imshow(circ, cmap=my_cmap,alpha=alpha,extent=extent,zorder=2)
    
    print(type(ax))
    if ax_tog==1:
        # Clip to make it circular
        print(centre )#+shift_centre*scale*[-1,-1]+scale/shape/2)
        patch = patches.Circle(centre +shift_centre*scale*[1,1], radius=clip, transform=ax.transData)
        im2.set_clip_path(patch)
        im1.set_clip_path(patch)
    
    if ax_tog==0:
        # Clip to make it circular
        patch = patches.Circle(centre +shift_centre*scale*[1,1]-scale/shape/4, radius=clip, transform=ax.transData)
        im2.set_clip_path(patch)
        im1.set_clip_path(patch)
        ax.axis('off')
        
def plot_2d_B(bx,by,mesh_params=None, ax=None,s=5,scale=7,mag_res=5, quiver=True, B_contour=True,phase=None,phase_res=np.pi/50):
    """ Plot projected B field
    quiver = turn on/off the arrows
    s = quiver density
    scale = quiver arrow size
    B_contour = turn on/off |B| contour lines
    mag_res = spacing of |B| contour lines in nT
    phase = pass phase shifts to plot phase contours
    phase_res = spacing of phase contours in radians
    """
    if ax == None:
        fig,ax = plt.subplots()
    
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(bx)
        p2 = (s[0],s[1],s[0])
        n = p2
        mesh_params = [p1,p2,n]
    
    p1,p2,n = mesh_params
    mag_B = np.hypot(bx,by)

    # Create alpha contour map
    my_cmap = alpha_cmap()

    # plot B field direction as a colour
    # using tan-1(vy/vx)
    angles = np.arctan2(by,bx)
    angles = shift_angles(angles,np.pi)
    ax.imshow(angles.T,origin='lower', 
               extent=[p1[0], p2[0], p1[1],p2[1]], cmap='hsv')

    # Plot magnitude of B as in black/transparent scale
    ax.imshow(mag_B.T,origin='lower', 
               extent=[p1[0], p2[0], p1[1],p2[1]],interpolation='spline16', cmap=my_cmap,alpha=1)

    ax.set_xlabel('x / m', fontsize = 16)
    ax.set_ylabel('y / m', fontsize = 16)
    
    # Quiver plot of Bx,By
    if quiver==True:
        x = np.linspace(p1[0],p2[0],num=n[0])
        y = np.linspace(p1[1],p2[1],num=n[1])
        xs,ys = np.meshgrid(x,y)
        ax.quiver(xs[::s,::s],ys[::s,::s],bx[::s,::s].T,by[::s,::s].T,color='white',scale=np.max(abs(mag_B))*scale,
                  pivot='mid',width=0.009,headaxislength=5,headwidth=4,minshaft=1.8)
    
    # Contour plot of |B|
    if B_contour==True:
        mag_range = (np.max(mag_B)-np.min(mag_B))/1e-9
        n_levels = int(mag_range/mag_res)
        cs = ax.contour(mag_B.T,origin='lower',levels=10, extent=[p1[0], p2[0], p1[1],p2[1]], alpha = .3,colors='white')
        
    # Contour plot of phase
    if type(phase)!=type(None):
        phase_range = (np.max(phase)-np.min(phase))/1e-9
        n_levels = int(phase_range/phase_res)
        cs = ax.contour(phase.T-np.min(phase).T,origin='lower',levels=10, extent=[p1[0], p2[0], p1[1],p2[1]], alpha = .3,colors='white')
        
def alpha_cmap():
    """ Returns a colormap object that is black,
    with alpha=1 at vmin and alpha=0 at vmax"""
    # Create a colour map which is just black
    colors = [(0.0, 'black'), (1.0, 'black')]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("alpha_cmap", colors)
    # Get colors from current map for values 1 to 256
    # These will all be black with alpha=1 (opaque), ie. [0,0,0,1]
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha (opaque and black (1) at vmin and fully transparanet at vmax)
    my_cmap[:,-1] = np.linspace(1,0,cmap.N)
    # create new colormap with the new alpha values
    my_cmap = ListedColormap(my_cmap)
    
    return my_cmap

def shift_angles(vals,angle=None):
    """ Takes angles currently in -pi to +pi range,
    and lets you shift them by an angle, keeping them
    in the same range."""
    
    if angle == None:
        return vals
    
    newvals = vals+angle
    for i,vv in enumerate(newvals):
        for j,v in enumerate(vv):
            if v > np.pi:
                newvals[i,j] = v - 2*np.pi
            if v < -np.pi:
                newvals[i,j] = v + 2*np.pi
            
    return newvals

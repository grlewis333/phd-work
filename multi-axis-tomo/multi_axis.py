import matplotlib.pyplot as plt                 # For normal plotting
from mpl_toolkits.mplot3d import proj3d         # For 3D plotting
import numpy as np                             # For maths
from scipy import ndimage                       # For image rotations
import RegTomoReconMulti as rtr                 # Modified version of Rob's CS code
import copy
import astra

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
    
    P = ndimage.rotate(P,ax,reshape=False,axes=(1,2))
    P = ndimage.rotate(P,ay,reshape=False,axes=(2,0))
    P = ndimage.rotate(P,az,reshape=False,axes=(0,1))

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

def rotation_matrix(ax,ay,az):
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
    
    mrot = mrotz.dot(mroty).dot(mrotx)
    
    return mrot

def get_astravec(ax,ay,az):
    """ Given angles in degrees, return r,d,u,v as a concatenation
    of four 3-component vectors"""
    # Due to indexing, ay needs reversing for desired behaviour
    ay = -ay
    
    # centre of detector
    d = [0,0,0]
    
    # 3D rotation matrix
    mrot = np.array(rotation_matrix(ax,ay,az))
    
    # ray direction r
    r = mrot.dot([0,0,1]) # think if *-1 is necessary
    # u (det +x)
    u = mrot.dot([1,0,0])
    # v (det +y)
    v = mrot.dot([0,1,0])

    return np.concatenate((r,d,u,v))

def generate_angles(x_tilt = (-70,70,11), y_tilt = None, n_random = 0):
    """ Return a list of [ax,ay,az] lists, each corresponding to axial
    rotations applied to [0,0,1] to get a new projection direction.
    
    To include tilt series about x or y, 
    specify _tilt with (min_angle, max_angle, n_angles) in deg for
    a linear spacing of angles, set to None if not desired. 
    
    Add n_random tilt orientations with angle chosen between +-90, 
    or set to 0 for none."""
    
    angles = []
    ax,ay,az = 0,0,0
    
    # x series
    if x_tilt != None:
        for ax in np.linspace(x_tilt[0],x_tilt[1],x_tilt[2]):
            angles.append([ax,ay,az])
    
    # y series
    ax,ay,az = 0,0,0
    if y_tilt != None:
        for ay in np.linspace(y_tilt[0],y_tilt[1],y_tilt[2]):
            angles.append([ax,ay,az])
    
    # random series
    if n_random > 0:
        for i in range(n_random):
            as_rand = np.random.rand(3)*180 - 90
            angles.append(as_rand.tolist())
    
    return angles

def generate_proj_data(P,angles):
    """ Returns projection dataset given phantom P
    and 3D projection angles list.
    
    Output is normalised and reshaped such that the
    projection slice dimension is in the middle, so as
    to be compatible with astra."""
    P_projs = []
    
    for [ax,ay,az] in angles:
        P_rot = rotate_bulk(P,ax,ay,az)
        P_rot_proj =np.flipud(np.mean(P_rot,axis=2).T)
        P_projs.append(P_rot_proj)
        
    # Prepare projections for reconstruction
    raw_data = np.array(P_projs)
    raw_data = raw_data -  raw_data.min()
    raw_data = raw_data/raw_data.max()
    raw_data = np.transpose(raw_data,axes=[1,0,2]) # reshape so z is middle column
        
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
                            balance = 1, steps = 'backtrack'):

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
        
        recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                balance=balance, steps=steps,
                callback=None)
    
    return recon

def reorient_reconstruction(r):
    # Swap columns back to match orientation of phantom
    r = np.transpose(r,[2,1,0]) # Reverse column order
    r = r[:,::-1,:] # Reverse the y data
    r = r -  r.min() # normalise
    r = r/r.max()

    recon_vector = copy.deepcopy(r)
    
    return recon_vector

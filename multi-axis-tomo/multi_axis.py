import matplotlib.pyplot as plt                 # For normal plotting
from mpl_toolkits.mplot3d import proj3d         # For 3D plotting
import numpy as np                             # For maths
from scipy import ndimage                       # For image rotations
import RegTomoReconMulti as rtr                 # Modified version of Rob's CS code
from scipy import optimize                      # For function minimization
import copy                                     # For deepcopy
import astra                                    # For tomography framework
import transforms3d                             # For some rotation work

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

def generate_angles(mode='x',n_tilt = 40, alpha=70,beta=40,gamma=180,dist_n2=5,tilt2='gamma'):
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
            az = 90
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
            az = 90
            for ax in np.linspace(-alpha,alpha,n_tilt/2):
                angles.append([ax,ay,az])
    
    if mode=='quad':
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
        
        if tilt2 == 'beta': 
            ay = np.linspace(-beta,beta,dist_n2)
            for x in ax:
                for y in ay:
                    angles.append([x,y,0])
        if tilt2 == 'gamma': 
            az = np.linspace(-gamma,gamma,dist_n2)
            for x in ax:
                for z in az:
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
                            balance = 1, steps = 'backtrack'):
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

def compare_recon_phantom(recon_vector,P,ax=0,ay=0,az=0):
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
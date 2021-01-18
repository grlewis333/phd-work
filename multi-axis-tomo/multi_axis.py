import matplotlib.pyplot as plt                 # For normal plotting
from mpl_toolkits.mplot3d import proj3d         # For 3D plotting
import numpy as np                             # For maths
from scipy import ndimage                       # For image rotations

print(99)
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

def plot_plane(ax,v=[0,0,1]):
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
    
def plot_both(P,ax,ay,az,save_path=None):
    # plot in 3D for a single tilt
    fig= plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    vx,vy,vz = angle_to_vector(ax,ay,az)
    Prot=rotate_bulk(P,ax,ay,az)
    plot_2d(X,Y,Z,Prot,ax=ax1,fig=fig)
    plot_plane(ax2,v=[vx,vy,vz])

    title = 'Projected density $(%i^{\circ},%i^{\circ},%i^{\circ})$' % (ax,ay,az)
    ax1.set_title(title,size=14)
    ax2.set_title('Projection direction visualised',size=14)
    
    if save_path != None:
        plt.savefig(save_path,bbox_inches='tight')
import numpy as np
def vortex(x,y):
    """ Returns mx and my components for a ring with 
    vortex state magnetisation given input x and y
    values that lie inside the ring material"""
    
    # gradient of tangent
    m = -x/y
    
    # angle between tangent and horizontal
    theta = np.arctan(m)
    
    # calculate relative cos/sin
        # for positive y
    if y > 0:
        C = np.cos(theta)
        S = np.sin(theta)
    
        # for y = 0
    elif y == 0:
        if x<0:
            C = 0
            S = 1
        else:
            C = 0
            S = -1
            
        # for negative y        
    else:
        C = -np.cos(theta)
        S = -np.sin(theta)
    
    return C, S

def onion(x,y):
    """ Returns mx and my components for a ring with 
    onion state magnetisation given input x and y
    values that lie inside the ring material"""
    m = -x/y
    theta = np.arctan(m)

    if y > 0:
        if x < 0:
            C = np.cos(theta)
            S = np.sin(theta)
        elif -2 < x < 2:
            C = 0
            S = 1
        else:
            C = -np.cos(theta)
            S = -np.sin(theta)

    elif y == 0:
        C = 0
        S = 1
            
    else:
        if x < 0:
            C = -np.cos(theta)
            S = -np.sin(theta)
        elif -2 < x < 2:
            C = 0
            S = 1
        else:
            C = np.cos(theta)
            S = np.sin(theta)
    return C, S
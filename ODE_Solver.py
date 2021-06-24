import numpy as np
import matplotlib.pyplot as plt
 
####CLEAN with_ASS and CLEAN EVERYTHING TOO, ALMOST DONE
####TRY ADDING ERROR ESTIMATES TO THIS

# k = slopes of y wr to x , m = slopes of z wr to x 
# x = 0 , y = 1 , z = 2

# x - N, y - phi, z - g or phi'
# phi' = g ; g' = (g^2/2 - 3)*(dV/dphi*1/V + g)
# phi = phi(N) ; V = V(phi)


def Update_vars(func, h, x_n , y_n , z_n):
    """Updation of variables

    Updates the variables x, y, z (y') in respect to the input function

    Args:
        func (function): This is the slope of z (y') with respect to x. 
        h (float): This is the step size from current value of x to its successive value
        x_n (float): Value of x after 'n' iterations
        y_n (float): Value of y after 'n' iterations
        z_n (float): Value of z(y') after 'n' iterations
    
    Returns:
        tuple: It contains the values of x, y and z (y') for the current iteration

    """
    m1 = func(x_n, y_n, z_n ) ; k1 = z_n
    m2 = func(x_n + h/2, y_n + h*k1/2, z_n + h*m1/2) ; k2 = z_n + h*m1/2
    m3 = func(x_n + h/2, y_n + h*k2/2, z_n + h*m2/2) ; k3 = z_n + h*m2/2
    m4 = func(x_n + h/2, y_n + h*k3/2, z_n + h*m3/2) ; k4 = z_n + h*m3

    y_n += h*(k1 + 2*k2 + 2*k3 + k4)/6
    z_n += h*(m1 + 2*m2 + 2*m3 + m4)/6
    x_n += h

    return x_n, y_n, z_n


def Solver(dzdx, x_i:float, x_f:float, y_i:float, dydx_i:float, h:float = 0.01):
    """ODE Solver

    Solves second order ODEs with Runge Kutta 4th order method

    Args:
        dzdx (function): This is the slope of z (y') with respect to x. 
        x_i (float): Initial value of x
        x_f (float): Final value of x
        y_i (float): Intial y condition
        dydx_i (float) : Initial y' condition 
        h (float): This is the step size from current value of x to its successive value 
    
    Returns:
        numpy.ndarray: It contains all the values of x, y and z (y') from all the iterations

    """
    vals = np.zeros( (3, int((x_f - x_i)/h)) ) 
    y_n, z_n = y_i, dydx_i

    for n, x_n in enumerate( np.arange(x_i, x_f, h) ):
        vals[0][n] , vals[1][n] , vals[2][n] = x_n , y_n , z_n

        x_n, y_n, z_n = Update_vars(dzdx, h, x_n, y_n, z_n)

    vals[0][-1], vals[1][-1], vals[2][-1] = x_n, y_n, z_n

    return vals


def Solver_with_AdSS(dzdx, x_i:float, x_f:float, y_i:float, dydx_i:float, h:float = 0.01, min_diff:float = 1e-5):
    """ODE Solver with Adaptive Step Sizing

    Solves ODE with Runge kutta 4th order method. Adaptive Step Sizing helps in speeding up ODEs with lot of flat slopes in its solution 
    and helps in collecting necessary datapoints, discarding fillers. It adapts it self to the curve by changing the step size appropriately. 
    Adaptive step sizing can slow down simple ODEs and ODEs with lot of steeper slopes but a trade off with rich data.

    Args:
        dzdx (function): This is the slope of z (y') with respect to x. 
        x_i (float): Initial value of x
        x_f (float): Final value of x
        y_i (float): Intial y condition
        dydx_i (float) : Initial y' condition 
        h (float): This is the step size from current value of x to its successive value 
    
    Other Parameters:
        min_diff (float): This is the minimum tolerance of difference of y between current step size and its half or double step size
    
    Returns:
        numpy.ndarray: It contains all the values of x, y and z (y') from all the iterations

    """
    h_min = h
    x_n, y_n , z_n = x_i, y_i , dydx_i
    vals = [[x_i],[y_i],[dydx_i]]

    while True:
        h_step = Update_vars(dzdx, h , x_n , y_n , z_n)
        half_h_step = Update_vars(dzdx, h/2 , x_n , y_n , z_n)
        double_h_step = Update_vars(dzdx, 2*h , x_n , y_n , z_n)

        if abs((h_step[1] - half_h_step[1])/h_step[1]) < min_diff :
            h /= 2
            x_n , y_n , z_n = half_h_step
        if abs((h_step[1] - double_h_step[1])/h_step[1]) < min_diff :
            h *= 2
            x_n , y_n , z_n = double_h_step
        else:
            x_n , y_n , z_n = h_step
            h = h_min
        
        vals[0].append(x_n)
        vals[1].append(y_n)
        vals[2].append(z_n)

        if x_n >= x_f:
            break

    return np.array(vals)


if __name__ == '__main__':
    slope = lambda x, y, z: x*np.sin(x)
    values = Solver(dzdx = slope, x_i = 0, x_f = 30, y_i = 1, dydx_i = 0, h = 0.001)
    print(f'number of datapoints : {values.shape[1]}')
    
    plt.rc('font', **{'family' : 'serif', 'size' : 15})
    plt.style.use('dark_background')

    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 5))

    ax[0].plot(values[0], values[1], c = 'c')
    ax[0].set_title("y vs x")

    ax[1].plot(values[0], values[2], c = 'r')
    ax[1].set_title("y' vs x")

    ax[2].plot(values[1], values[2], c = 'b')
    ax[2].set_title("y' vs y")

    plt.tight_layout()
    plt.show()
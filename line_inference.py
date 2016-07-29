import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a = 0.0
b = 100.0
n = 101

c = 30.0
r = 10.0

nn = np.linspace(a, b, n)

cc = 1.0/(b-a)*np.ones(n)

h_fig, h_ax = plt.subplots(1,1)
h_cc, = h_ax.plot(nn, cc, 'r-')

p_mat = np.array([[.9, .1], [.05, .95]])

def in_range(x,cc):
    if abs(x-cc) < r:
        return 1
    else:
        return 0

def z(x):
    rx = np.random.rand(1)
    rr = p_mat[in_range(x,c),1]
    if rx < rr:
        return True
    return False            
    
def init():
    cc = (b-a)/n*np.ones(n)
    h_ax.plot([c-r, c+r], [.3, .3], c='blue', linewidth=2)
    return h_cc,

#plt.show()
symbols = ('r^','go')

def animate(i, *args, **kwargs):
    x = a + np.random.rand(1)*(b-a)
    z_x = z(x)
    h_ax.plot(x, 0.3, symbols[z_x])
    for i in range(len(cc)):
        cc[i] = p_mat[in_range(x,nn[i]), z_x]*cc[i]
    ccs = cc.sum()/(b-a)
    for i in range(len(cc)):
        cc[i] = cc[i]/ccs
    h_cc.set_ydata(cc)
    h_ax.set_ylim(cc.min(), min(5.0, cc.max()))
    return h_cc,

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = 100, interval = 100, blit = True, repeat = False)
ani.save('line_vid.mp4', writer = 'avconv', fps=10, bitrate=5000)

h_fig.show()
import numpy as np
import matplotlib.pyplot as plt
import mcmc
import matplotlib.animation as animation
#plt.style.use('ggplot')

def gauss_test(x):
    return np.exp(-(x[0]+2)**2/2.0 - (x[1]+2)**2/4) + np.exp(-(x[0]-1)**2/4.0 - (x[1]-2)**2/5.0) # x[0]+5.0

def lin_test(x):
    return x[0]+5.1


class GridCost:
    def __init__(self, npfile):
        self.cost = np.load(npfile)
        assert len(self.cost.shape) is 2

    def cost_fun(self, x):
        return self.cost[int(x[0]), int(x[1])]


grid_cost = GridCost('../data/esdf.npy')

test_fun = grid_cost.cost_fun       # gauss_test

# Sample
lims = np.array([[0, 0], [grid_cost.cost.shape[0], grid_cost.cost.shape[1]]])
ns = mcmc.MCMCSampler(test_fun, lims)

fig,ax = plt.subplots()
fig.set_size_inches((7,7))
cmap = plt.cm.jet
cmap.set_bad(color='white')
gcost = np.ma.masked_where(grid_cost.cost <= 0, grid_cost.cost)

ax.imshow(gcost.transpose(), cmap=cmap, origin='lower')
n_samples = 5000
burn_in = 0
X,fX = ns.sample_chain(n_samples, burn_in, X_start = [10, 10])

samples, = ax.plot(X[0:1, 0], X[0:1, 1], '.k')
line, = ax.plot(X[0:1, 0], X[0:1, 1], 'r-')

def animate(i):
    samples.set_xdata(X[0:i, 0])  # update the data
    samples.set_ydata(X[0:i, 1])
    mm = max(0, i - 200)
    line.set_xdata(X[mm:i, 0])  # update the data
    line.set_ydata(X[mm:i, 1])
    return [line, samples]

# Init only required for blitting to give a clean slate.
def init():
    line.set_xdata(X[0:1, 0])  # reset
    line.set_ydata(X[0:1, 1])
    samples.set_xdata(X[0:1, 0])  # update the data
    samples.set_ydata(X[0:1, 1])
    return [line, samples]


ani = animation.FuncAnimation(fig, animate, n_samples, init_func=init,
                              interval=25, blit=True)
ani.save('../vid/mcmc_sampling.ogg',
           writer='ffmpeg', fps=10, codec='libtheora', bitrate=8000)
plt.show()
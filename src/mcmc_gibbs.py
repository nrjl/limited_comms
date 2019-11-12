import numpy as np
import matplotlib.pyplot as plt
import mcmc
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
# plt.style.use('ggplot')
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)
sigma = [1.5, 1.0]
mu = [2.0, 2.0]
rho = 0.7


x = np.linspace(mu[0]-4*sigma[0], mu[0]+4*sigma[0], 101)
y = np.linspace(mu[1]-4*sigma[1], mu[1]+4*sigma[1], 101)
xx, yy = np.meshgrid(x, y)
pos = np.empty(xx.shape + (2,))
pos[:, :, 0] = xx; pos[:, :, 1] = yy
rv = multivariate_normal(mu, [[sigma[0]**2, rho*sigma[0]*sigma[1]],
                                      [rho*sigma[0]*sigma[1], sigma[1]**2]])

# Gibbs sampler
n_samples = 1000
X = np.zeros((n_samples, 2))
X[0] = [1.5, 1.8]
cdim = 0
for i in range(n_samples-1):
    odim = 1-cdim
    X[i+1] = X[i]
    X[i+1,cdim] = np.random.normal(
        mu[cdim]+sigma[cdim]/sigma[odim]*rho*(X[i,odim]-mu[odim]),
        (1-rho**2)*sigma[cdim]**2, 1)
    cdim = odim

# plt.contourf(xx, yy, rv.pdf(pos), 20)

fig,ax = plt.subplots()
fig.set_size_inches((7,5))
plt.imshow(rv.pdf(pos).T, origin='lower', extent=(x[0],x[-1],y[0],y[-1]),
           aspect='equal', cmap=plt.cm.viridis)
line, = ax.plot(X[0:1, 0], X[0:1, 1], color='grey')
samples, = ax.plot(X[0:1, 0], X[0:1, 1], '.w')
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")

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
ani.save('../vid/mcmc_gibbs/%03d.png', writer='imagemagick')
# ani.save('../vid/mcmc_gibbs.ogg',
#            writer='ffmpeg', fps=10, codec='libtheora', bitrate=8000)
plt.show()
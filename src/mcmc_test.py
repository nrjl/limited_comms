import numpy as np
import matplotlib.pyplot as plt
import mcmc
plt.style.use('ggplot')

def gauss_test(x):
    return np.exp(-(x[0]+2)**2/2.0 - (x[1]+2)**2/4) + np.exp(-(x[0]-1)**2/4.0 - (x[1]-2)**2/5.0) # x[0]+5.0

def lin_test(x):
    return x[0]+5.1

test_fun = gauss_test

# Sample
lims = np.array([[-5.0, -5.0], [5.0, 5.0]])
ns = mcmc.MCMCSampler(test_fun, lims)

xx = np.linspace(lims[0,0], lims[1,0], 500)
yy = np.linspace(lims[0,1], lims[1,1], 500)
zz = np.zeros((len(xx), len(yy)))
for i,x in enumerate(xx):
    for j,y in enumerate(yy):
        zz[i,j] = test_fun([x,y])

grid_integral = zz.sum()*ns.lim_range.prod()/np.prod(zz.shape)

fig,ax = plt.subplots()
ax.imshow(zz.transpose(), origin='lower', extent=[lims[0,1], lims[1,1], lims[0,0], lims[1,0]])

n_samples = 5000
burn_in = 1000
X,fX = ns.sample_chain(n_samples, burn_in, X_start = [-4.0, -4.0])
chain_integral = (1.0/((fX**-1).sum()/n_samples))*ns.lim_range.prod()
ax.plot(X[:,0], X[:,1], '.k-')
plt.show()

rho_R = 0.1
depth = 3
cR = 1.0
mchain_integral = 1.0
nc_samples = n_samples/depth
for i in range(depth):
    crho = cR*rho_R
    cR = cR - crho
    nt = mcmc.MCMCSampler(lambda x: (test_fun(x)**(crho)), lims)
    Xt,fXt = nt.sample_chain(nc_samples, burn_in)
    mchain = 0.0
    for j in range(Xt.shape[0]):
        mchain += test_fun(Xt[j])**cR
    mchain = mchain/nc_samples
    mchain_integral *= mchain
    print "Nested: <F^({0})>_({1}) = {2}".format(cR, crho, mchain)
    cR = crho
mchain_integral *= ns.lim_range.prod()
print "Grid integral: {0}".format(grid_integral)
print "Chain integral: {0}".format(chain_integral)
print "Nested integral: {0}".format(mchain_integral)
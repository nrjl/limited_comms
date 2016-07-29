import numpy as np
import matplotlib.pyplot as plt
import sensor_models
import belief_state
import mcmc
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
np.random.seed(2)

# Truth data
field_size = (100,100)
target_centre = np.array([62.0,46.0])
target_radius = 15.0

# Observation model
obs_model = sensor_models.logistic_obs
obs_kwargs = {'r':target_radius,'true_pos':0.95,'true_neg':0.90}

# Observation MCMC sampler
mcmc_n_samples = 100000
mcmc_burnin = 1000

# Observation set
evidence_x = sensor_models.arc_samples((46.0,13.0), 30.0, 160, 50, n=20)

# Other constants
obs_symbols = ['r^','go']

fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
test_state = belief_state.belief(field_size, obs_model, obs_kwargs, ax=ax)
obs = test_state.generate_observations(evidence_x,c=target_centre)
test_state.add_observations(obs)

pc = test_state.centre_probability_map(1000)
ax[0].imshow(pc.transpose(), origin='lower',vmin=0)

pz = test_state.observation_probability_map(1000)
ax[1].imshow(pz.transpose(), origin='lower',vmin=0)

Vx = test_state.utility_map(1000)
ax[2].imshow(Vx.transpose(), origin='lower',vmin=0)

for axx in ax:
    axx.plot(target_centre[0],target_centre[1],'wx',mew=2,ms=10)
    for xx,zz in obs:
        axx.plot(xx[0],xx[1],obs_symbols[zz])
    
fig.set_size_inches(10,4)
ax[0].set_xlim(-.5, field_size[0]-0.5)
ax[0].set_ylim(-.5, field_size[1]-0.5)
ax[0].set_title('$p(c|I)$')
ax[1].set_title(r'$p(z=T|I)$')
ax[2].set_title(r'$V(x) = \mathbb{E}_Z \left[ D_{KL}\left(p(\cdot|z(x),I)||p(\cdot|I)\right) \right]$')



plt.show()


## SCRAP
## MCMC sampler to sample from p_z_given_c
#obs_lims = [[-field_size[0],-field_size[1]],[field_size[0],field_size[1]]]
#mcmc_obsF_obj = mcmc.MCMCSampler(obs_model, obs_lims, func_kwargs=dict(obs_kwargs, z=False, c=(0,0)))
#mcmc_obsFX,mcmc_obsFp = mcmc_obsF_obj.sample_chain(mcmc_n_samples, mcmc_burnin)
#mcmc_obsT_obj = mcmc.MCMCSampler(obs_model, obs_lims, func_kwargs=dict(obs_kwargs, z=True, c=(0,0)))
#mcmc_obsTX,mcmc_obsTp = mcmc_obsT_obj.sample_chain(mcmc_n_samples, mcmc_burnin)
#np.random.shuffle(mcmc_obsFX)
#np.random.shuffle(mcmc_obsTX)
#
#fig2,ax2 = plt.subplots()
#ax2.plot(mcmc_obsFX[::100,0],mcmc_obsFX[::100,1],obs_symbols[0])
#ax2.plot(mcmc_obsTX[::100,0],mcmc_obsTX[::100,1],obs_symbols[1])

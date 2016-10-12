import numpy as np
import matplotlib.pyplot as plt
import pickle

filelist = ['no_share.pkl','20pc_share.pkl','10pc_share.pkl']

trial_steps = [] #np.zeros((100,len(filelist)))
n_shares = [] #np.zeros((100,len(filelist)))

for ii,ff in enumerate(filelist):
    with open('../data/'+ff,'rb') as fh:
        trial_steps.append(pickle.load(fh))
        n_shares.append(pickle.load(fh))

trial_steps = np.array(trial_steps).transpose()
n_shares = np.array(n_shares).transpose()
hfig,hax = plt.subplots()
hax.boxplot(trial_steps)
hax.set_title('Raw observation count means')

hf2,hax2 = plt.subplots()
obs_diffs = (trial_steps[:,1:].transpose()-trial_steps[:,0]).transpose()
hax2.boxplot(obs_diffs)
hax2.set_title('Observation count differences')
plt.show()

# hfig.savefig('../fig/bad_sensor_obscount.pdf',bbox_inches='tight')
# hf2.savefig('../fig/bad_sensor_obsdiff.pdf',bbox_inches='tight')

print 'Value of sharing (no. observations): {0}'.format((-obs_diffs/n_shares[:,1:]).mean(axis=0))
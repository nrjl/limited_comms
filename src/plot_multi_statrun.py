import numpy as np
import matplotlib.pyplot as plt
import pickle

badfilelist = ['badno_share.pkl','bad20pc_share.pkl','bad10pc_share.pkl']
goodfilelist = ['no_share.pkl','20pc_share.pkl','10pc_share.pkl']

trial_steps = [] #np.zeros((100,len(filelist)))
n_shares = [] #np.zeros((100,len(filelist)))

for ff in badfilelist:
    with open('../data/'+ff,'rb') as fh:
        trial_steps.append(pickle.load(fh))
        n_shares.append(pickle.load(fh))

trial_steps = np.array(trial_steps).transpose()
n_shares = np.array(n_shares).transpose()
hfig,hax = plt.subplots()
hax.boxplot(trial_steps)
hax.set_xticklabels(['No sharing', r'$20\%$ Max. KLD', r'$10\%$ Max. KLD'])
hax.set_xlabel('Sharing threshold')
hax.set_ylabel('Number of observations to find target')
#hax.set_title('Target search with noisy sensor')

hf2,hax2 = plt.subplots()
obs_diffs = (trial_steps[:,1:].transpose()-trial_steps[:,0]).transpose()
hax2.boxplot(obs_diffs)
hax2.set_title('Observation count differences')

def get_means_std(flist):
    means, stds, shares = [],[],[]
    for ff in flist:
        with open('../data/'+ff,'rb') as fh:
            steps = pickle.load(fh)
            means.append(np.mean(steps))
            stds.append(np.std(steps))
            shares.append(np.mean(pickle.load(fh)))
    return means,stds,shares
    
def share_labels(ax, rects, vals):
    # attach some text labels
    alim = ax.get_ylim()
    vpos = alim[0]+(alim[1]-alim[0])*0.05
    for rect,val in zip(rects,vals):
        ax.text(rect.get_x() + rect.get_width()/2., vpos,
                "{0:0.1f}".format(val),
                ha='center', va='bottom')
            
good_means, good_stds, good_shares = get_means_std(goodfilelist)
bad_means, bad_stds, bad_shares = get_means_std(badfilelist)
hf3,hax3 = plt.subplots()
width = 0.35       # the width of the bars
ind = np.arange(len(good_means))
rects1 = hax3.bar(ind, good_means, width, color='cornflowerblue', yerr=good_stds)
rects2 = hax3.bar(ind + width, bad_means, width, color='sandybrown', yerr=bad_stds)
hax3.set_ylabel('Number of observations to find target')
hax3.set_xticks(ind + width)
hax3.set_xticklabels(['No sharing', r'$20\%$ Max. KLD', r'$10\%$ Max. KLD'])
hax3.set_xlim([0,ind[-1]+2*width])
share_labels(hax3, rects1, good_shares)
share_labels(hax3, rects2, bad_shares)
hax3.legend((rects1[0], rects2[0]), ('High-quality sensor', 'Low-quality sensor'))


# hfig.savefig('../fig/bad_sensor_obscount.pdf',bbox_inches='tight')
# hf2.savefig('../fig/bad_sensor_obsdiff.pdf',bbox_inches='tight')
# hf3.savefig('../fig/sensor_counts.pdf',bbox_inches='tight')

plt.show()

print 'Value of sharing (no. observations): {0}'.format((-obs_diffs/n_shares[:,1:]).mean(axis=0))
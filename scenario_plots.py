import numpy as np
import matplotlib.pyplot as plt
import fields

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

# Hard coded for paper figures
def paper_samples(c, r, t1, t2, n=6):
    ang = np.linspace(t1*np.pi/180, t2*np.pi/180,n)
    x = c[0]+r*np.cos(ang)
    y = c[1]+r*np.sin(ang)
    return np.vstack((x,y)).transpose()

true_pos = 0.95
true_neg = 0.90
def logistic_obsT(x,c,r):
    return true_pos - (true_pos+true_neg-1.0)/(1+np.exp(-0.25*(np.linalg.norm(x-c)-r)))
logistic_obs = [lambda x,c,r : 1.0-logistic_obsT(x,c,r), logistic_obsT]

obs_fun = logistic_obs  # step_obs

paperx = 140
papery = 100
papertarget = [62,46]
paperradius = 15
z1_pos = paper_samples([50,40],45, 160, 120)
z1 = np.array([0,0,0,0,1,0])
z2_pos = paper_samples([46,13],30, 110, 50)
z2 = np.array([0,0,0,1,1,0])
z3_pos = paper_samples([94,40],40, -30, 30)
z3 = np.array([0,0,0,0,0,0])
z0_pos = np.vstack((z1_pos[0:2,:],z2_pos[0:2,:],z3_pos[0:2,:]))
z0 = np.hstack((z1[0:2],z2[0:2],z3[0:2]))
zA_pos = np.vstack((z0_pos,z1_pos[2:,:]))
zA = np.hstack((z0,z1[2:]))
zB_pos = np.vstack((z0_pos,z2_pos[2:,:]))
zB = np.hstack((z0,z2[2:]))
zC_pos = np.vstack((z0_pos,z3_pos[2:,:]))
zC = np.hstack((z0,z3[2:]))

#plt.show()
symbols = ('r^','go')

hTf, hTa = plt.subplots()
pt_field = fields.TrueField(paperx, papery, obs_fun, obs_fun_kwargs={'c': papertarget, 'r': paperradius}, ax=hTa)


def show_evidence(nx,ny,obs_fun,obs_fun_kwargs,zp,z):
    hf, ha = plt.subplots()
    ha.plot(papertarget[0],papertarget[1],'wx',mew=2,ms=10)
    p_field = fields.ProbField(nx, ny, obs_fun, obs_fun_kwargs=obs_fun_kwargs, ax=ha)
    for i,z_pos in enumerate(zp):
        p_field.update_field(z_pos, z[i],fmax=0.00075)
        ha.plot(z_pos[0],z_pos[1],symbols[z[i]])
    return hf,ha,p_field

h0f,h0a,pf0 = show_evidence(paperx, papery, obs_fun,{'r': paperradius}, z0_pos,z0)
hAf,hAa,pfa = show_evidence(paperx, papery, obs_fun,{'r': paperradius}, zA_pos,zA)
hBf,hBa,pfb = show_evidence(paperx, papery, obs_fun,{'r': paperradius}, zB_pos,zB)
hCf,hCa,pfc = show_evidence(paperx, papery, obs_fun,{'r': paperradius}, zC_pos,zC)
h1f,h1a,pf1 = show_evidence(paperx, papery, obs_fun,{'r': paperradius}, np.vstack((z1_pos,z2_pos,z3_pos)), np.hstack((z1,z2,z3)))

#h0f.savefig('fig/k0b.png', bbox_inches='tight')
#hAf.savefig('fig/kAb.png', bbox_inches='tight')
#hBf.savefig('fig/kBb.png', bbox_inches='tight')
#hCf.savefig('fig/kCb.png', bbox_inches='tight')
#h1f.savefig('fig/k1b.png', bbox_inches='tight')
plt.show()
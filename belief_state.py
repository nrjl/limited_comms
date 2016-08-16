import numpy as np
import sensor_models

class belief:
    def __init__(self, gridsize, p_z_given_x, p_z_given_x_kwargs = {}, pose=np.array([0.0,0.0,0.0]), unshared_obs = False, motion_model=None):
        self.nx = gridsize[0]
        self.ny = gridsize[1]
        self.x = np.arange(self.nx)
        self.y = np.arange(self.ny)
        self.p_uniform = 1.0/np.prod(gridsize)
        self.p_z_given_x = p_z_given_x
        self.p_z_given_x_kwargs = p_z_given_x_kwargs
        self.p_c = self.uniform_prior
        self.reset_observations()
        self.pose = pose
        self.update_pc_map = False
        self.unshared_obs = unshared_obs
        self.set_motion_model(motion_model)
        
    def set_current_pose(self, pose):
        self.pose = pose
    
    def get_current_pose(self):
        return self.pose
        
    def set_motion_model(self,motion_model):
        self.motion_model = motion_model
        
    def generate_observations(self,x,c):
        obs=[]
        for xx in x:
            p_T = self.p_z_given_x(xx,True,c,**self.p_z_given_x_kwargs)
            zz = np.random.uniform()<p_T
            obs.append((xx,zz))
        return obs
    
    def add_observations(self,obs):
        self.observations.extend(obs)
        for ii,xc in enumerate(self.csamples):
            self.pIgc[ii] *= self.p_I_given_c(c=xc,obs=obs)
                
        if self.update_pc_map:
            for obx,obz in obs:
                # For each added observation, update p(I|c) for the pc map samples
                for ii,xx in enumerate(self.x):
                    for jj,yy in enumerate(self.y):
                        self.pIgc_map[ii,jj] *= self.p_z_given_x(obx,obz,c=[xx,yy],**self.p_z_given_x_kwargs)            
                
    def get_observations(self):
        return self.observations
        
    def reset_observations(self):
        self.observations = []
        if hasattr(self, 'csamples'):
            self.pIgc = np.array([self.p_I_given_c(xc) for xc in self.csamples])
        
    def uniform_prior(self, c):
        return self.p_uniform
        
    def assign_prior_sample_set(self, xs):
        # This takes a set of samples drawn from the prior over centre p(c)
        # and calculates the evidence likelihood for each of them
        self.csamples = xs
        self.pIgc = np.array([self.p_I_given_c(xc) for xc in xs])
            
    def uniform_prior_sampler(self,n=1):
        xs = np.random.uniform(high=self.nx,size=(n,1))
        ys = np.random.uniform(high=self.ny,size=(n,1))
        return np.hstack((xs,ys))    
        
    def mc_p_I(self,xs=None):
        if xs is not None:
            mc_accumulator = 0.0
            for xc in xs:
                mc_accumulator += self.p_I_given_c(c=xc)
            return mc_accumulator/xs.shape[0]
        return self.pIgc.mean()
    
    def p_I_given_c(self,c,obs=None):
        p_evidence = 1.0
        if obs is None:
            obs=self.observations
        for xx,zz in obs:
            p_evidence *= self.p_z_given_x(xx,zz,c=c,**self.p_z_given_x_kwargs)
        return p_evidence
    
    def p_c_given_I(self,c,pc=None,pI=None):
        pIgc = self.p_I_given_c(c)
        if pc is None:
            pc = self.p_c(c)
        if pI is None:
            pI = self.mc_p_I()
        return pIgc*pc/pI
    
    def pzgc_samples(self,x,z):
        return np.array([self.p_z_given_x(x,z,c=xc,**self.p_z_given_x_kwargs) for xc in self.csamples])
                
    def mc_p_z_given_I(self,x,z,xs=None,pzgc=None):
        if xs is None:
            xs = self.csamples
            pIgc = self.pIgc
        else:
            pIgc = np.array([self.p_I_given_c(xc) for xc in xs])
        
        if pzgc is None:
            pzgc = np.array([self.p_z_given_x(x,z,c=xc,**self.p_z_given_x_kwargs) for xc in xs])
        
        pz_accumulator = np.multiply(pIgc,pzgc).sum()
        # pI = pI_accumulator/xs.shape[0], and likewise for the pz_accumlator,
        # so p(z(x)|I) is just the ratio (the counts will cancel)
        return pz_accumulator/pIgc.sum()
        
    def kld_utility(self,x):
        # Ignoring p(I)
        pc = self.p_c(0) # Only for uniform
        pzgc_a = self.pzgc_samples(x,True)
        pzgI = self.mc_p_z_given_I(x,True,pzgc=pzgc_a)
        
        VT_accumulator = 0.0
        VF_accumulator = 0.0
        for ii,xc in enumerate(self.csamples):
            pzgc = pzgc_a[ii]
            pIgc = self.pIgc[ii]
            VT_accumulator += pzgc*pIgc*pc*np.log(pzgc/pzgI)
            VF_accumulator += (1-pzgc)*pIgc*pc*np.log((1.0-pzgc)/(1.0-pzgI))
        return VT_accumulator+VF_accumulator
        
    def kld(self,X,Z):
        pzgc = np.array([self.pzgc_samples(x,z) for x,z in zip(X,Z)])
        p_new_obs_c = np.product(pzgc, axis=0)
        p_all_samples = p_new_obs_c*self.pIgc
        p_all = p_all_samples.mean()
        pI = self.mc_p_I()
        d_kl = 1/p_all*np.mean(p_all_samples*np.log(p_new_obs_c*pI/p_all))
        return d_kl

    def centre_probability_map(self):
        pcI = np.zeros((self.nx,self.ny))
        pI = self.mc_p_I()
        pc = self.p_c(0) # Constant (uniform) prior
        for ii,xx in enumerate(self.x):
            for jj,yy in enumerate(self.y):
                pcI[ii,jj] = self.p_c_given_I(c=[xx,yy],pc=pc,pI=pI)
        return pcI
        
    def observation_probability_map(self):
        pz = np.zeros((self.nx,self.ny))
        for ii,xx in enumerate(self.x):
            for jj,yy in enumerate(self.y):
                pz[ii,jj] = self.mc_p_z_given_I([xx,yy],True)
        return pz    
    
    def utility_map(self,mcn=1000):
        Vx = np.zeros((self.nx,self.ny))
        for ii,xx in enumerate(self.x):
            for jj,yy in enumerate(self.y):
                Vx[ii,jj] = self.kld_utility([xx,yy])
        return Vx

    def select_observation(self,target_centre):
        future_pose = self.motion_model.get_end_points(self.get_current_pose())
        Vx = np.array([self.kld_utility(pos[0:2]) for pos in future_pose])
        amax = np.argmax(Vx)
        
        next_pose = future_pose[amax]
        self.set_current_pose(next_pose)
        cobs = self.generate_observations([next_pose[0:2]],c=target_centre)
        self.add_observations(cobs)
        return future_pose,Vx,amax
        
    def persistent_centre_probability_map(self):
        pI = self.mc_p_I()
        pc = self.p_c(0)
        if self.update_pc_map == False:
            self.update_pc_map = True
            self.pIgc_map = self.centre_probability_map()*pI/pc
        return self.pIgc_map*pc/pI
        
#for obx,obz in obs:
#            # For each added observation, update p(I|c) for the c samples
#            for ii,xc in enumerate(self.csamples):
#                self.pIgc[ii] = self.pIgc[ii]*self.p_z_given_x(obx,obz,c=xc,**self.p_z_given_x_kwargs)
#        

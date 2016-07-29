import numpy as np
import sensor_models

class belief:
    def __init__(self, gridsize, p_z_given_x, p_z_given_x_kwargs = {}):
        self.nx = gridsize[0]
        self.ny = gridsize[1]
        self.x = np.arange(self.nx)
        self.y = np.arange(self.ny)
        self.p_uniform = 1.0/(gridsize[0]*gridsize[1])
        self.p_z_given_x = p_z_given_x
        self.p_z_given_x_kwargs = p_z_given_x_kwargs
        self.p_c = self.uniform_prior
        self.observations = []
        
    def generate_observations(self,x,c):
        obs=[]
        for xx in x:
            p_T = self.p_z_given_x(xx,True,c,**self.p_z_given_x_kwargs)
            zz = np.random.uniform()<p_T
            obs.append((xx,zz))
        return obs
    
    def add_observations(self,obs):
        self.observations.extend(obs)
        for obx,obz in obs:
            # For each added observation, update p(I|c) for the c samples
            for ii,xc in enumerate(self.csamples):
                self.pIgc[ii] = self.pIgc[ii]*self.p_z_given_x(obx,obz,c=xc,**self.p_z_given_x_kwargs)            
        
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
        if obs==None:
            obs=self.observations
        for xx,zz in obs:
            p_evidence *= self.p_z_given_x(xx,zz,c=c,**self.p_z_given_x_kwargs)
        return p_evidence
    
    def p_c_given_I(self,c,pc=None,pI=None):
        pIgc = self.p_I_given_c(c)
        if pc == None:
            pc = self.p_c(c)
        if pI == None:
            pI = self.mc_p_I()
        return pIgc*pc/pI
        
    def mc_p_z_given_I(self,x,z,xs=None):
        if xs is None:
            xs = self.csamples
            pIgc = self.pIgc
        else:
            pIgc = np.array([self.p_I_given_c(xc) for xc in xs])
            
        pzgc = np.array([self.p_z_given_x(x,z,c=xc,**self.p_z_given_x_kwargs) for xc in xs])
        
        pz_accumulator = np.multiply(pIgc,pzgc).sum()
        # pI = pI_accumulator/xs.shape[0], and likewise for the pz_accumlator,
        # so p(z(x)|I) is just the ratio (the counts will cancel)
        return pz_accumulator/pIgc.sum()
        
    def kld_utility(self,x):
        # Ignoring p(I)
        pc = self.p_c(0) # Only for uniform
        pzgI = self.mc_p_z_given_I(x,True)
        
        VT_accumulator = 0.0
        VF_accumulator = 0.0
        for ii,xc in enumerate(self.csamples):
            pzgc = self.p_z_given_x(x,True,c=xc,**self.p_z_given_x_kwargs)
            pIgc = self.pIgc[ii]
            VT_accumulator += pzgc*pIgc*pc*np.log(pzgc/pzgI)
            VF_accumulator += (1-pzgc)*pIgc*pc*np.log((1.0-pzgc)/(1.0-pzgI))
        return VT_accumulator+VF_accumulator            
        
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
    
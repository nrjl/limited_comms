import numpy as np
import sensor_models

class belief:
    def __init__(self, gridsize, p_z_given_x, p_z_given_x_kwargs = {}, ax=None):
        self.nx = gridsize[0]
        self.ny = gridsize[1]
        self.x = np.arange(self.nx)
        self.y = np.arange(self.ny)
        self.p_uniform = 1.0/(gridsize[0]*gridsize[1])
        self.p_z_given_x = p_z_given_x
        self.p_z_given_x_kwargs = p_z_given_x_kwargs
        self.p_c = self.uniform_prior
        self.ax = ax
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
        
    def uniform_prior(self, c):
        return self.p_uniform
            
    def uniform_prior_sampler(self,n=1):
        xs = np.random.uniform(high=self.nx,size=(n,1))
        ys = np.random.uniform(high=self.ny,size=(n,1))
        return np.hstack((xs,ys))    
        
    def mc_p_I(self,xs):
        mc_accumulator = 0.0
        for xc in xs:
            mc_accumulator += self.p_I_given_c(c=xc)
        return mc_accumulator/xs.shape[0]
    
    def p_I_given_c(self,c):
        p_evidence = 1.0
        for xx,zz in self.observations:
            p_evidence *= self.p_z_given_x(xx,zz,c=c,**self.p_z_given_x_kwargs)
        return p_evidence
    
    def p_c_given_I(self,c,xs,pc=None,pI=None):
        pIgc = self.p_I_given_c(c)
        if pc == None:
            pc = self.p_c(c)
        if pI == None:
            pI = self.mc_p_I(xs)
        return pIgc*pc/pI
        
    def mc_p_z_given_I(self,x,z,n_pIsamples=None,xs=None):
        # Generate pI:
        if [n_pIsamples, xs].count(None) != 1:
            raise TypeError("Specify number of samples or sample list")
        if xs is None:
            xs = self.uniform_prior_sampler(n_pIsamples)

        pI_accumulator = 0.0
        pz_accumulator = 0.0
        for xc in xs:
            pIgc = self.p_I_given_c(xc)
            pI_accumulator += pIgc
            pz_accumulator += pIgc*self.p_z_given_x(x,z,c=xc,**self.p_z_given_x_kwargs)
        # pI = pI_accumulator/xs.shape[0], and likewise for the pz_accumlator,
        # so p(z(x)|I) is just the ratio (the counts will cancel)
        return pz_accumulator/pI_accumulator
        
    def kld_utility(self,x,xs):
        # Ignoring p(I)
        pc = self.p_c(0) # Only for uniform
        pzgI = self.mc_p_z_given_I(x,True,xs=xs)
        
        VT_accumulator = 0.0
        VF_accumulator = 0.0
        for xc in xs:
            pzgc = self.p_z_given_x(x,True,c=xc,**self.p_z_given_x_kwargs)
            pIgc = self.p_I_given_c(xc)
            VT_accumulator += pzgc*pIgc*pc*np.log(pzgc/pzgI)
            VF_accumulator += (1-pzgc)*pIgc*pc*np.log((1.0-pzgc)/(1.0-pzgI))
        return VT_accumulator+VF_accumulator            
        
    def centre_probability_map(self,mcn=1000):
        pcI = np.zeros((self.nx,self.ny))
        xs = self.uniform_prior_sampler(mcn)
        pI = self.mc_p_I(xs)
        pc = self.p_c(0) # Constant (uniform) prior
        for ii,xx in enumerate(self.x):
            for jj,yy in enumerate(self.y):
                pcI[ii,jj] = self.p_c_given_I(c=[xx,yy],xs=xs,pc=pc,pI=pI)
        return pcI
        
    def observation_probability_map(self,mcn=1000):
        xs = self.uniform_prior_sampler(mcn)
        pz = np.zeros((self.nx,self.ny))
        for ii,xx in enumerate(self.x):
            for jj,yy in enumerate(self.y):
                pz[ii,jj] = self.mc_p_z_given_I([xx,yy],True,xs=xs)
        return pz    
    
    def utility_map(self,mcn=1000):
        xs = self.uniform_prior_sampler(mcn)
        Vx = np.zeros((self.nx,self.ny))
        for ii,xx in enumerate(self.x):
            for jj,yy in enumerate(self.y):
                Vx[ii,jj] = self.kld_utility([xx,yy],xs)
        return Vx
    
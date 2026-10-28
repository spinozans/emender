"""Compact full covariance CMA-ES, Hansen rank-one/rank-mu equations; no dependency."""
import numpy as np

class CMAES:
    def __init__(self, mean, sigma=.2, population=12, seed=4201):
        self.mean = np.array(mean, dtype=float)
        self.n = n = len(mean)
        self.sigma = sigma
        self.population = population
        self.mu = population//2
        w = np.log(self.mu+.5)-np.log(np.arange(1,self.mu+1))
        self.w = w/w.sum()
        self.mueff = me = 1/(self.w@self.w)
        self.cc = (4+me/n)/(n+4+2*me/n)
        self.cs = (me+2)/(n+me+5)
        self.c1 = 2/((n+1.3)**2+me)
        self.cm = min(1-self.c1,2*(me-2+1/me)/((n+2)**2+me))
        self.damp = 1+2*max(0,np.sqrt((me-1)/(n+1))-1)+self.cs
        self.chi = np.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
        self.C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.rng = np.random.default_rng(seed)
        self.generation = 0

    def ask(self):
        vals, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(vals,1e-20))
        self.y = (self.rng.standard_normal((self.population,self.n))*self.D)@self.B.T
        self.x = self.mean+self.sigma*self.y
        return self.x.copy()

    def tell(self, fitness):
        fitness = np.asarray(fitness)
        if fitness.shape != (self.population,) or not np.isfinite(fitness).all():
            raise ValueError('finite population fitness required')
        y = self.y[np.argsort(fitness,kind='stable')[:self.mu]]
        yw = self.w@y
        self.mean += self.sigma*yw
        inv = (self.B/self.D)@self.B.T
        self.ps = (1-self.cs)*self.ps+np.sqrt(self.cs*(2-self.cs)*self.mueff)*(inv@yw)
        self.generation += 1
        hs = np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.generation))/self.chi < 1.4+2/(self.n+1)
        self.pc = (1-self.cc)*self.pc+hs*np.sqrt(self.cc*(2-self.cc)*self.mueff)*yw
        self.C = ((1-self.c1-self.cm)*self.C+self.c1*(np.outer(self.pc,self.pc)+(1-hs)*self.cc*(2-self.cc)*self.C)
                  +self.cm*(y.T*self.w)@y)
        self.C = (self.C+self.C.T)/2
        self.sigma *= np.exp(self.cs/self.damp*(np.linalg.norm(self.ps)/self.chi-1))

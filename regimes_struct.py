import numpy as np
from scipy.optimize import linear_sum_assignment
import mvlib_web as W

def rel(tt,lam,th,lm):
    r=th.shape[1]; Tt=tt.transpose(1,0,2).reshape(r,-1); Te=th.transpose(1,0,2).reshape(r,-1)
    C=((Tt[:,None]-Te[None])**2).sum(-1)+(lam[:,None]-lm[None])**2
    _,p=linear_sum_assignment(C)
    return float(np.linalg.norm(Tt-Te[p])/max(np.linalg.norm(Tt),1e-30))

def controlled(sep,sig,seed=0):
    B=W.Basis(K=10,N=801); xg=B.xg; dx=B.dx; r=2; d=2
    u=np.zeros((d,r,B.N))
    rg=np.random.default_rng(seed)
    for j in range(d):
        for m in range(r):
            c=0.5+(m-0.5)*0.35
            u[j,m]=0.5*np.exp(-(xg-(c-sep/2))**2/(2*sig**2))+0.5*np.exp(-(xg-(c+sep/2))**2/(2*sig**2))
            u[j,m]/=u[j,m].sum()*dx
    return B,u,np.array([0.5,0.5])

def run(name,kind,n,d=2,sep=None,sig=None,contam=0.0,seeds=4):
    mom=[]; em=[]; st=[]
    for s in range(seeds):
        B=W.Basis(K=10,N=801)
        if kind=="controlled":
            B,u,lam=controlled(sep,sig,s)
        else:
            u,lam=W.make_u_mix(B,kind,d,2,2,seed=s)
        tt=W.theta_true_of(u,B)
        _,X=W.sample(u,lam,B,n,seed=s+100,contam=contam)
        th,lm=W.moments_theta(X,B,2,d,1e-4,5,s); mom.append(rel(tt,lam,th,lm))
        the,lme=W.em_fit(X,B,2,L=1,d=d,n_restarts=4,seed=s); em.append(rel(tt,lam,the,lme))
        ls,ws,xs,vs=W._struct_init(th,lm,B,d,2,2)
        ls,ws,xs,vs=W._struct_em(X,ls,ws,xs,vs,d,2,2,T=10)
        ths=W._struct_theta(ls,ws,xs,vs,B,d,2,2); st.append(rel(tt,lam,ths,ls))
    print(f"{name}: mom={np.mean(mom):.3f} em={np.mean(em):.3f} struct={np.mean(st):.3f}")

run("overlapping", "controlled", 1000, sep=0.10, sig=0.06)
run("separated", "controlled", 1000, sep=0.45, sig=0.035)
run("beta1000", "beta", 1000)
run("beta3000", "beta", 3000)
run("heavy3000", "laplace", 3000)
run("contam0.2", "gauss", 1000, contam=0.2)
run("d3beta", "beta", 1000, d=3)

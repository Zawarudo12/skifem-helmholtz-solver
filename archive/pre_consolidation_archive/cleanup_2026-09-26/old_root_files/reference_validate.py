from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

@dataclass(frozen=True)
class Cfg:
    wavelength: float = 0.633
    n0: complex = 1.0+0j
    n1: complex = 1.7+0j
    n2: complex = 1.0+0j
    air_top: float = 0.50
    slab_d: float = 0.50
    air_bot: float = 0.50
    @property
    def k0(self): return 2*np.pi/self.wavelength
    @property
    def y0(self): return self.air_top
    @property
    def y1(self): return self.air_top+self.slab_d
    @property
    def L(self): return self.air_top+self.slab_d+self.air_bot


def airy(cfg: Cfg):
    n0,n1,n2=cfg.n0,cfg.n1,cfg.n2
    d=cfg.slab_d; k0=cfg.k0
    r01=(n0-n1)/(n0+n1)
    r12=(n1-n2)/(n1+n2)
    t01=2*n0/(n0+n1)
    t12=2*n1/(n1+n2)
    delta=k0*n1*d
    den=1+r01*r12*np.exp(2j*delta)
    r=(r01+r12*np.exp(2j*delta))/den
    t=t01*t12*np.exp(1j*delta)/den
    R=abs(r)**2
    T=(np.real(n2)/np.real(n0))*abs(t)**2
    return r,t,float(R),float(T)


def exact_field(y, cfg: Cfg):
    y=np.asarray(y,float)
    r,t,_,_=airy(cfg)
    k0=cfg.k0; n0,n1,n2=cfg.n0,cfg.n1,cfg.n2
    a=cfg.y0; d=cfg.slab_d
    # Interface-referenced field has unit incident amplitude at y=a.
    A=0.5*((1+r)+(n0/n1)*(1-r))
    B=0.5*((1+r)-(n0/n1)*(1-r))
    phase_to_interface=np.exp(1j*k0*n0*a)
    z=y-a
    u=np.empty(y.shape,dtype=complex)
    top=y<=a
    slab=(y>a)&(y<a+d)
    bot=y>=a+d
    u[top]=phase_to_interface*(np.exp(1j*k0*n0*z[top])+r*np.exp(-1j*k0*n0*z[top]))
    u[slab]=phase_to_interface*(A*np.exp(1j*k0*n1*z[slab])+B*np.exp(-1j*k0*n1*z[slab]))
    u[bot]=phase_to_interface*t*np.exp(1j*k0*n2*(z[bot]-d))
    return u


def build_edges(cfg: Cfg, h_target: float):
    segs=[(0,cfg.y0),(cfg.y0,cfg.y1),(cfg.y1,cfg.L)]
    parts=[]
    for j,(a,b) in enumerate(segs):
        n=max(1,int(np.ceil((b-a)/h_target)))
        yy=np.linspace(a,b,n+1)
        if j: yy=yy[1:]
        parts.append(yy)
    return np.concatenate(parts)


def solve_fem(cfg: Cfg, h_target: float, order: int):
    edges=build_edges(cfg,h_target)
    ne=len(edges)-1
    if order==1:
        coords=edges.copy()
        conn=np.column_stack([np.arange(ne),np.arange(1,ne+1)])
    elif order==2:
        coords=np.empty(2*ne+1)
        coords[0::2]=edges
        coords[1::2]=0.5*(edges[:-1]+edges[1:])
        conn=np.column_stack([2*np.arange(ne),2*np.arange(ne)+1,2*np.arange(ne)+2])
    else: raise ValueError
    N=len(coords)
    A=lil_matrix((N,N),dtype=complex)
    for e,(yl,yr) in enumerate(zip(edges[:-1],edges[1:])):
        h=yr-yl; ym=0.5*(yl+yr)
        n=cfg.n1 if cfg.y0 < ym < cfg.y1 else (cfg.n0 if ym<cfg.y0 else cfg.n2)
        if order==1:
            Ke=np.array([[1,-1],[-1,1]],float)/h
            Me=h*np.array([[2,1],[1,2]],float)/6
        else:
            Ke=np.array([[7,-8,1],[-8,16,-8],[1,-8,7]],float)/(3*h)
            Me=h*np.array([[4,2,-1],[2,16,2],[-1,2,4]],float)/30
        Ae=Ke-cfg.k0**2*(n**2)*Me
        ids=conn[e]
        for i,I in enumerate(ids):
            for j,J in enumerate(ids): A[I,J]+=Ae[i,j]
    # Robin on top and bottom: du/dn - i k0 u = g; matrix contribution -i k0.
    A[0,0] += -1j*cfg.k0*cfg.n0
    A[-1,-1] += -1j*cfg.k0*cfg.n2
    b=np.zeros(N,dtype=complex)
    b[0] = -2j*cfg.k0*cfg.n0  # incident exp(+i k y), y_top=0, outward normal=-y
    u=spsolve(csc_matrix(A),b)
    return coords, conn, edges, u


def eval_fem(yq, coords, conn, edges, u, order):
    yq=np.asarray(yq)
    idx=np.searchsorted(edges,yq,side='right')-1
    idx=np.clip(idx,0,len(edges)-2)
    yl=edges[idx]; yr=edges[idx+1]; h=yr-yl
    xi=2*(yq-yl)/h-1
    if order==1:
        N1=(1-xi)/2; N2=(1+xi)/2
        return N1*u[conn[idx,0]]+N2*u[conn[idx,1]]
    N1=0.5*xi*(xi-1); N2=1-xi**2; N3=0.5*xi*(xi+1)
    return N1*u[conn[idx,0]]+N2*u[conn[idx,1]]+N3*u[conn[idx,2]]


def rel_l2(cfg, coords, conn, edges, u, order):
    gx,gw=np.polynomial.legendre.leggauss(10)
    num=den=0.0
    for e,(yl,yr) in enumerate(zip(edges[:-1],edges[1:])):
        y=0.5*(yl+yr)+0.5*(yr-yl)*gx
        uh=eval_fem(y,coords,conn,edges,u,order)
        ue=exact_field(y,cfg)
        jac=0.5*(yr-yl)
        num += np.sum(gw*np.abs(uh-ue)**2)*jac
        den += np.sum(gw*np.abs(ue)**2)*jac
    return np.sqrt(num/den)


def fit_rt(cfg, coords, conn, edges, u, order):
    yt=np.linspace(0.12*cfg.air_top,0.82*cfg.air_top,80)
    yb=np.linspace(cfg.y1+0.18*cfg.air_bot,cfg.L-0.12*cfg.air_bot,80)
    ut=eval_fem(yt,coords,conn,edges,u,order)
    ub=eval_fem(yb,coords,conn,edges,u,order)
    kt=cfg.k0*cfg.n0; kb=cfg.k0*cfg.n2
    Xt=np.column_stack([np.exp(1j*kt*yt),np.exp(-1j*kt*yt)])
    Xb=np.column_stack([np.exp(1j*kb*yb),np.exp(-1j*kb*yb)])
    ai,ar=np.linalg.lstsq(Xt,ut,rcond=None)[0]
    at,ain_bottom=np.linalg.lstsq(Xb,ub,rcond=None)[0]
    R=abs(ar/ai)**2
    T=(np.real(cfg.n2)/np.real(cfg.n0))*abs(at/ai)**2
    return float(R),float(T),abs(ain_bottom/ai)

if __name__=='__main__':
    cfg=Cfg()
    ra,ta,Ra,Ta=airy(cfg)
    print('analytic',Ra,Ta,Ra+Ta,ra,ta)
    for order in (1,2):
        print('order',order)
        hs=[]; es=[]
        for ppw in [6,8,12,16,24,32,48]:
            h=(cfg.wavelength/np.real(cfg.n1))/ppw
            coords,conn,edges,u=solve_fem(cfg,h,order)
            err=rel_l2(cfg,coords,conn,edges,u,order)
            R,T,leak=fit_rt(cfg,coords,conn,edges,u,order)
            hmax=np.max(np.diff(edges))
            hs.append(hmax); es.append(err)
            print(ppw, len(u), hmax, err, R, T, R+T, leak)
        # fit last 4 points
        slope=np.polyfit(np.log(hs[-4:]),np.log(es[-4:]),1)[0]
        print('slope',slope)

# Plot/table helper for the executed independent 1-D reference FEM.
def write_validation_outputs(outdir='/mnt/data/phase1_helmholtz/results_reference'):
    from pathlib import Path
    import csv
    import matplotlib.pyplot as plt
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    cfg=Cfg()
    rows=[]
    curves={}
    for order in (1,2):
        hs=[]; errs=[]
        for ppw in [6,8,12,16,24,32,48]:
            h=(cfg.wavelength/np.real(cfg.n1))/ppw
            coords,conn,edges,u=solve_fem(cfg,h,order)
            err=rel_l2(cfg,coords,conn,edges,u,order)
            R,T,leak=fit_rt(cfg,coords,conn,edges,u,order)
            hmax=float(np.max(np.diff(edges)))
            hs.append(hmax); errs.append(err)
            rows.append([order,ppw,len(u),hmax,err,R,T,R+T,leak])
        slope=float(np.polyfit(np.log(hs[-4:]),np.log(errs[-4:]),1)[0])
        curves[order]=(np.array(hs),np.array(errs),slope)
    with (out/'error_table.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['order','ppw_in_slab','dofs_1d','hmax_um','rel_L2','R','T','R_plus_T','bottom_incoming_ratio']); w.writerows(rows)

    plt.figure(figsize=(6.4,4.6))
    for order,(hs,errs,slope) in curves.items():
        plt.loglog(hs,errs,'o-',label=f'P{order}, fitted slope={slope:.3f}')
    plt.gca().invert_xaxis(); plt.xlabel('element size h [um]'); plt.ylabel('relative L2 field error'); plt.legend(); plt.tight_layout(); plt.savefig(out/'convergence_reference.png',dpi=180); plt.close()

    # Fine P2 field profile.
    h=(cfg.wavelength/np.real(cfg.n1))/32
    coords,conn,edges,u=solve_fem(cfg,h,2)
    y=np.linspace(0,cfg.L,1800); uh=eval_fem(y,coords,conn,edges,u,2); ue=exact_field(y,cfg)
    plt.figure(figsize=(7.2,4.5)); plt.plot(y,np.real(ue),label='analytic Re(Ez)'); plt.plot(y,np.real(uh),'--',label='reference FEM P2 Re(Ez)'); plt.plot(y,np.abs(ue),label='analytic |Ez|',alpha=.8); plt.plot(y,np.abs(uh),'--',label='reference FEM P2 |Ez|',alpha=.8); plt.axvspan(cfg.y0,cfg.y1,alpha=.12,label='slab'); plt.xlabel('depth y [um]'); plt.ylabel('field amplitude'); plt.legend(ncol=2,fontsize=8); plt.tight_layout(); plt.savefig(out/'field_profile_reference.png',dpi=180); plt.close()

    # Analytic R/T sweep and exact FP resonance check.
    wavelengths=np.linspace(.45,.95,501); Rs=[]; Ts=[]
    for lam in wavelengths:
        c=Cfg(wavelength=float(lam)); _,_,R,T=airy(c); Rs.append(R); Ts.append(T)
    m=np.arange(1,8); res=2*np.real(cfg.n1)*cfg.slab_d/m; res=res[(res>=wavelengths.min())&(res<=wavelengths.max())]
    checks=[]
    for lam in res:
        c=Cfg(wavelength=float(lam)); _,_,R,T=airy(c); checks.append((lam,R,T,R+T))
    with (out/'fabry_perot_check.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['lambda_um','R','T','R_plus_T']); w.writerows(checks)
    plt.figure(figsize=(7.2,4.5)); plt.plot(wavelengths,Rs,label='analytic R'); plt.plot(wavelengths,Ts,label='analytic T');
    for rr in res: plt.axvline(rr,lw=.8,alpha=.35)
    plt.xlabel('vacuum wavelength [um]'); plt.ylabel('power fraction'); plt.ylim(-.03,1.05); plt.legend(); plt.tight_layout(); plt.savefig(out/'rt_analytic_reference.png',dpi=180); plt.close()
    return rows,curves,checks

if __name__=='__main__':
    rows,curves,checks=write_validation_outputs()
    print('executed slopes:', {k:v[2] for k,v in curves.items()})
    print('FP checks:', checks)

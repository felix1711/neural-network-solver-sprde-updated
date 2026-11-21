import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def exact(u, e):
    return (np.exp(-(1-u)/np.sqrt(e))+np.exp(-u/np.sqrt(e)))/(1+np.exp(-1/np.sqrt(e))) - np.cos(np.pi*u)**2

class PINN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
        for m in self.net:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self,u):
        return u*(1-u)*self.net(u)

def rhs(u,e):
    return -torch.cos(torch.pi*u)**2 - 2*e*torch.pi**2*torch.cos(2*torch.pi*u)

def loss_fn(model,u,e):
    x=u.detach().clone().requires_grad_(True)
    v=model(x)
    vu=torch.autograd.grad(v,x,torch.ones_like(v),create_graph=True)[0]
    vuu=torch.autograd.grad(vu,x,torch.ones_like(vu),create_graph=True)[0]
    return ((-e*vuu+v-rhs(x,e))**2).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epsilon',type=float,default=1e-3)
    ap.add_argument('--epochs',type=int,default=3000)
    ap.add_argument('--target-loss',type=float,default=1e-4)
    ap.add_argument('--points',type=int,default=2000)
    ap.add_argument('--adam-lr',type=float,default=5e-4)
    ap.add_argument('--lbfgs-every',type=int,default=100)
    ap.add_argument('--lbfgs-max-iter',type=int,default=20)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_default_dtype(torch.float64)
    e=a.epsilon; model=PINN()
    u=torch.linspace(0,1,a.points).reshape(-1,1)
    adam=optim.Adam(model.parameters(),lr=a.adam_lr)
    lbfgs=optim.LBFGS(model.parameters(),lr=.5,max_iter=a.lbfgs_max_iter,history_size=50,line_search_fn='strong_wolfe')
    hist=[]
    for epoch in range(1,a.epochs+1):
        adam.zero_grad(); loss=loss_fn(model,u,e); loss.backward(); adam.step()
        if epoch%a.lbfgs_every==0:
            def closure():
                lbfgs.zero_grad(); l=loss_fn(model,u,e); l.backward(); return l
            lbfgs.step(closure)
        if epoch==1 or epoch%10==0:
            val=loss_fn(model,u,e).item(); hist.append((epoch,val))
            if epoch%100==0 or val<a.target_loss: print(f'Epoch {epoch}/{a.epochs}: {val:.8e}')
            if val<a.target_loss:
                print('Target PDE residual reached.'); break
    out=Path(__file__).resolve().parents[1]/'results'/'test_problem_1'; out.mkdir(parents=True,exist_ok=True)
    ut=torch.linspace(0,1,2000).reshape(-1,1)
    with torch.no_grad(): pred=model(ut).numpy()
    un=ut.numpy(); truth=exact(un,e); err=np.abs(pred-truth)
    mae=float(err.mean()); mx=float(err.max()); rel=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)); final=float(loss_fn(model,u,e).item())
    print(f'PDE residual={final:.8e} | MAE={mae:.8e} | Max error={mx:.8e} | Relative L2={rel:.8e}')
    np.savetxt(out/'loss_history.csv',np.array(hist),delimiter=',',header='epoch,pde_residual',comments='')
    (out/'metrics.txt').write_text(f'epsilon={e}\nfinal_pde_residual={final}\nMAE={mae}\nmax_error={mx}\nrelative_L2={rel}\n')
    torch.save(model.state_dict(),out/'model_state.pt')
    plt.figure(figsize=(8,5)); plt.plot(un,truth,label='Analytical solution'); plt.plot(un,pred,'--',label='PINN prediction'); plt.xlabel('u'); plt.ylabel('v(u)'); plt.title('Test Problem 1: Analytical vs PINN'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(out/'solution_comparison.png',dpi=300); plt.close()
    plt.figure(figsize=(8,5)); plt.plot(un,err); plt.xlabel('u'); plt.ylabel('Absolute error'); plt.title('Test Problem 1: Absolute Error'); plt.grid(); plt.tight_layout(); plt.savefig(out/'absolute_error.png',dpi=300); plt.close()
    ep,ls=np.array(hist).T; plt.figure(figsize=(8,5)); plt.semilogy(ep,ls); plt.xlabel('Epoch'); plt.ylabel('PDE residual'); plt.title('Test Problem 1: Adam-LBFGS convergence'); plt.grid(); plt.tight_layout(); plt.savefig(out/'training_loss.png',dpi=300); plt.close()

if __name__=='__main__': main()

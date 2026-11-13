"""Shared, training-fitted fixed normalization; no test/validation statistics."""
import numpy as np
import torch


def fit_transform(data,mode):
    u,_,y=data['train']; device=u.device
    x=u.detach().cpu().numpy().astype(np.float64)
    mean=x.mean(0,keepdims=True)
    centered=x-mean; covariance=centered.T@centered/len(x)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    scale=float(y.double().square().mean().sqrt())
    if mode=='none':
        matrix=np.eye(x.shape[1]);mean=np.zeros_like(mean);scale=1.
    elif mode=='standard':
        matrix=np.diag(1/np.maximum(np.sqrt(np.diag(covariance)),1e-6))
    elif mode=='whiten':
        ridge=1e-3*float(eigenvalues[-1])
        matrix=(eigenvectors*(np.maximum(eigenvalues,0)+ridge)**-.5)@eigenvectors.T
    else:raise ValueError(mode)
    scale=max(scale,1e-12)
    info=dict(mode=mode,mean=mean,matrix=matrix,target_scale=np.array(scale),
              covariance_eigenvalues=eigenvalues,time_scale=np.array(1. if mode=='none' else 2/3),
              time_offset=np.array(0. if mode=='none' else -1.))
    if mode=='none':return data,info
    mu=torch.tensor(mean,dtype=torch.float32,device=device)
    transform=torch.tensor(matrix,dtype=torch.float32,device=device)
    transformed={split:((u-mu)@transform,2*t/3-1,y/scale) for split,(u,t,y) in data.items()}
    return transformed,info

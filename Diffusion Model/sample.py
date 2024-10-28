import torch

from u_net import Unet
from loss_func import *
import cv2
from tqdm import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):#传入的x是x_t
    t=torch.tensor([t]).to('cuda')
    coeff=((1-alphas[t]).to('cuda')/one_minus_alphas_bar_sqrt[t].to('cuda')).to('cuda')
    eps_theta=model(x,t)
    #eps_theta=eps_theta.view(3,64,64)
    mean = (1 / (alphas[t]).sqrt().to('cuda')) * (x - (coeff * eps_theta))
    # mean=(1/(1-betas[t]).sqrt().to('cuda'))*(x-(coeff*eps_theta))
    z=torch.randn_like(x).to('cuda')
    sigma_t=betas[t].sqrt().to('cuda')
    sample=mean+sigma_t*z
    return sample


def p_sample_loop(model,n_steps,betas,one_minus_alphas_bar_sqrt):
    cur_x=torch.randn(1,3,64,64).to('cuda')
    for i in tqdm(reversed(range(n_steps))):
    # for i in reversed(range(n_steps)):
        cur_x=p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)


    return cur_x.view(3,64,64)

# def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):#传入的x是x_t
#     t=torch.tensor([t])
#     coeff=((1-alphas[t])/one_minus_alphas_bar_sqrt[t])
#     eps_theta=model(x,t)
#     #eps_theta=eps_theta.view(3,64,64)
#     mean = (1 / (alphas[t]).sqrt()) * (x - (coeff * eps_theta))
#     # mean=(1/(1-betas[t]).sqrt().to('cuda'))*(x-(coeff*eps_theta))
#     z=torch.randn_like(x)
#     sigma_t=betas[t].sqrt()
#     sample=mean+sigma_t*z
#     return sample
#
#
# def p_sample_loop(model,n_steps,betas,one_minus_alphas_bar_sqrt):
#     cur_x=torch.randn(1,3,64,64)
#     for i in tqdm(reversed(range(n_steps))):
#     # for i in reversed(range(n_steps)):
#         cur_x=p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)


    # return cur_x.view(3,64,64)
model=torch.load("checkpoint/state_model epoch 805.pt")
model.eval()
img=p_sample_loop(model,time_steps,betas,one_minus_alphas_bar_sqrt)
img=img.detach().cpu().numpy().transpose(1,2,0)
img=(img+1.0) * 255 / 2
img=Image.fromarray(np.uint8(img))


fig =plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
plt.imshow(img)
plt.show()
# img=Image.fromarray(np.uint8(img))
# cv2.imwrite('/root/autodl-tmp/img1.jpg', img)
# cv2.imwrite('E:/img.jpg', img)


# fig =plt.figure(figsize=(5,5))
# ax=fig.add_subplot(111)
# plt.imshow(img)
# plt.show()
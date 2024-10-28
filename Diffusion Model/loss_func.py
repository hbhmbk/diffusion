from u_net import *
import torch
from torch.nn import functional as F
time_steps=500




def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
betas=linear_beta_schedule(time_steps)


# betas=torch.linspace(-10,10,time_steps)
# betas=torch.sigmoid(betas)*(0.5e-2-1e-5)+1e-5
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)


def _extract( a, t, x_shape):
    # get the param of given timestep t
    batch_size = t.shape[0]
    out = a.gather(0, t).float()
    print(out.shape)
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


def q_t(x_0, t):
    noise = torch.randn_like(x_0)
    alpha_t = alphas_bar_sqrt[t]
    alpha_1_m_t = one_minus_alphas_bar_sqrt[t]
    return alpha_t * x_0 + alpha_1_m_t * noise

def loss_fun(model,device,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,time_steps):
    device = device
    model = model.to(device)
    batch_size = x_0.shape[0]
    t = torch.randint(0, time_steps, size=(batch_size // 2,))
    t = torch.cat([t, time_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1).to(device)
    # print(t)
    a = alphas_bar_sqrt[t]
    # print(a.shape)
    a = a[:, None, None].to(device)

    aml = one_minus_alphas_bar_sqrt[t]
    aml = aml[:, None, None].to(device)
    e = torch.randn_like(x_0).to(device)
    x = (x_0 * a + e * aml)

    output = model(x, t.squeeze(-1))

    loss = F.mse_loss(e, output)

    return loss

if __name__ == '__main__':

    # model=Unet(8,with_time_emb=True,use_convnext=False).to('cuda')
    x_0=torch.randn(10,3,64,64)
    # # loss=loss_fun(model,'cuda',x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,200)
    # shape=x_0.shape
    # batch_size = x_0.shape[0]
    # t = torch.randint(0, time_steps, size=(batch_size // 2,))
    # t = torch.cat([t, time_steps - 1 - t], dim=0)
    # print(t)
    # print(betas)
    # t = t.unsqueeze(-1)
    # a = alphas_bar_sqrt[t]
    # a = a[:, None, None]
    # t=t.reshape(10)
    # a=torch.randn(10,1)
    # i=torch.randn(10,3,64,64)
    # # b=a*i
    # a = a[:, None, None]
    # c=a*i
    # print(a.shape)
    # out=_extract(alphas_bar_sqrt,t,x_0.shape)
    # print(a==out)




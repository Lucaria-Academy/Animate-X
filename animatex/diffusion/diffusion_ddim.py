import torch
import math

from utils.registry_class import DIFFUSION
from .schedules import beta_schedule, sigma_schedule
from typing import Callable, List, Optional
import numpy as np

def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t].view(shape).to(x)

@DIFFUSION.register_class()
class DiffusionDDIMSR(object):
    def __init__(self, reverse_diffusion, forward_diffusion, **kwargs):
        from .diffusion_gauss import GaussianDiffusion
        self.reverse_diffusion = GaussianDiffusion(sigmas=sigma_schedule(reverse_diffusion.schedule, **reverse_diffusion.schedule_param), 
                                                   prediction_type=reverse_diffusion.mean_type)
        self.forward_diffusion = GaussianDiffusion(sigmas=sigma_schedule(forward_diffusion.schedule, **forward_diffusion.schedule_param), 
                                                   prediction_type=forward_diffusion.mean_type)


@DIFFUSION.register_class()
class DiffusionDPM(object):
    def __init__(self, forward_diffusion, **kwargs):
        from .diffusion_gauss import GaussianDiffusion
        self.forward_diffusion = GaussianDiffusion(sigmas=sigma_schedule(forward_diffusion.schedule, **forward_diffusion.schedule_param),
            prediction_type=forward_diffusion.mean_type) 


@DIFFUSION.register_class()
class DiffusionDDIM(object):
    def __init__(self,
                 schedule='linear_sd',
                 schedule_param={},
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon = 1e-12,
                 rescale_timesteps=False,
                 noise_strength=0.0, 
                 **kwargs):
        
        assert mean_type in ['x0', 'x_{t-1}', 'eps', 'v']
        assert var_type in ['learned', 'learned_range', 'fixed_large', 'fixed_small']
        assert loss_type in ['mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1','charbonnier']
        
        betas = beta_schedule(schedule, **schedule_param)
        assert min(betas) > 0 and max(betas) <= 1

        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)

        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type # eps
        self.var_type = var_type # 'fixed_small'
        self.loss_type = loss_type # mse
        self.epsilon = epsilon # 1e-12
        self.rescale_timesteps = rescale_timesteps # False
        self.noise_strength = noise_strength # 0.0

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], alphas.new_zeros([1])])
        
        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
    

    def sample_loss(self, x0, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            if self.noise_strength > 0:
                b, c, f, _, _= x0.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=x0.device)
                noise = noise + self.noise_strength * offset_noise
        return noise


    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0).
        """
        # noise = torch.randn_like(x0) if noise is None else noise
        noise = self.sample_loss(x0, noise)
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
               _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0).
        """
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var
    
    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var
    
    @torch.no_grad()
    def p_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * torch.exp(0.5 * log_var) * noise
        return xt_1, x0
    
    @torch.no_grad()
    def p_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1).
        """
        # prepare input
        b = noise.size(0)
        xt = noise
        
        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale)
        return xt
    
    def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        # print("============")
        # print(model_kwargs) # 'pose_embedding'
        # print("============")
        # predict distribution
        # print("=============p_mean_variance============")
        # print("=============xt.shape============", xt.shape)
        # print("=============t.shape============", t.shape)
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2

            # print("===============model_kwargs[0]==============")

            # print(model_kwargs[0])

            # print("===============model_kwargs[0]==============")

            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0])

            # print("===============model_kwargs[1]==============")

            # print(model_kwargs[1])

            # print("===============model_kwargs[1]==============")

            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1])
            dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
            out = torch.cat([
                u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]),
                y_out[:, dim:]], dim=1) # guide_scale=9.0
        
        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)
        
        # compute mean and x0
        if self.mean_type == 'x_{t-1}':
            mu = out  # x_{t-1}
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                 _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
        elif self.mean_type == 'x0':
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'v':
            x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        
        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    @torch.no_grad()
    def ddim_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps
        
        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0
    
    @torch.no_grad()
    def ddim_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        from tqdm import tqdm
        for step in tqdm(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta)
            # from ipdb import set_trace; set_trace()
        return xt
    
    @torch.no_grad()
    def ddim_reverse_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas_next = _i(
            torch.cat([self.alphas_cumprod, self.alphas_cumprod.new_zeros([1])]),
            (t + stride).clamp(0, self.num_timesteps), xt)
        
        # reverse sample
        mu = torch.sqrt(alphas_next) * x0 + torch.sqrt(1 - alphas_next) * eps
        return mu, x0
    
    @torch.no_grad()
    def ddim_reverse_sample_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        # prepare input
        b = x0.size(0)
        xt = x0

        # reconstruction steps
        steps = torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp, percentile, guide_scale, ddim_timesteps)
        return xt
    
    @torch.no_grad()
    def plms_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
                eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            return eps
        
        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0
        
        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1] + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2] - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    @torch.no_grad()
    def plms_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // plms_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        eps_cache = []
        for step in steps:
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, plms_timesteps, eps_cache)
            
            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
        #return t.float()






@DIFFUSION.register_class()
class DiffusionDDIMLong(object):
    def __init__(self,
                 schedule='linear_sd',
                 schedule_param={},
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon = 1e-12,
                 rescale_timesteps=False,
                 noise_strength=0.0, 
                 **kwargs):

        assert mean_type in ['x0', 'x_{t-1}', 'eps', 'v']
        assert var_type in ['learned', 'learned_range', 'fixed_large', 'fixed_small']
        assert loss_type in ['mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1','charbonnier']
        
        betas = beta_schedule(schedule, **schedule_param)
        assert min(betas) > 0 and max(betas) <= 1

        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)

        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type # v
        self.var_type = var_type # 'fixed_small'
        self.loss_type = loss_type # mse
        self.epsilon = epsilon # 1e-12
        self.rescale_timesteps = rescale_timesteps # False
        self.noise_strength = noise_strength 

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], alphas.new_zeros([1])])
        
        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)
    

    def sample_loss(self, x0, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            if self.noise_strength > 0:
                b, c, f, _, _= x0.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=x0.device)
                noise = noise + self.noise_strength * offset_noise
        return noise


    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0).
        """
        # noise = torch.randn_like(x0) if noise is None else noise
        noise = self.sample_loss(x0, noise)
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
               _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0).
        """
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var
    
    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var
    
    @torch.no_grad()
    def p_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * torch.exp(0.5 * log_var) * noise
        return xt_1, x0
    
    @torch.no_grad()
    def p_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1).
        """
        # prepare input
        b = noise.size(0)
        xt = noise
        
        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale)
        return xt
    
    def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, context_size=32, context_stride=1, context_overlap=4, context_batch_size=1):
        r"""Distribution of p(x_{t-1} | x_t).
        """




        noise = xt
        context_queue = list(
                    context_scheduler(
                        0,
                        31,
                        noise.shape[2],
                        context_size=context_size,
                        context_stride=1,
                        context_overlap=4,
                    )
                )
        context_step = min(
                    context_stride, int(np.ceil(np.log2(noise.shape[2] / context_size))) + 1
                )
        # replace the final segment to improve temporal consistency
        num_frames = noise.shape[2]
        context_queue[-1] = [
                e % num_frames
                for e in range(num_frames - context_size * context_step, num_frames, context_step)
            ]
        
        import math
        # context_batch_size = 1
        num_context_batches = math.ceil(len(context_queue) / context_batch_size)
        global_context = []
        for i in range(num_context_batches):
            global_context.append(
                context_queue[
                    i * context_batch_size : (i + 1) * context_batch_size
                ]
            )
        noise_pred = torch.zeros_like(noise)
        noise_pred_uncond = torch.zeros_like(noise)
        counter = torch.zeros(
                    (1, 1, xt.shape[2], 1, 1),
                    device=xt.device,
                    dtype=xt.dtype,
                )
        
        for i_index, context in enumerate(global_context):
            
            
            latent_model_input = torch.cat([xt[:, :, c] for c in context])
            bs_context = len(context)
            
            # print("model_kwargs[0][local_image].shape: ", model_kwargs[0]['local_image'].shape)
            # print("model_kwargs[0][dwpose].shape: ", model_kwargs[0]['dwpose'].shape)
            # print("model_kwargs[0][pose_embedding].shape: ", model_kwargs[0]['pose_embedding'].shape)

            model_kwargs_new = [{
                                    'y': None,
                                    "local_image": None if not model_kwargs[0].__contains__('local_image') else torch.cat([model_kwargs[0]["local_image"][:, :, c] for c in context]),
                                    'image':  None if not model_kwargs[0].__contains__('image') else model_kwargs[0]["image"].repeat(bs_context, 1, 1),
                                    'dwpose':  None if not model_kwargs[0].__contains__('dwpose') else torch.cat([model_kwargs[0]["dwpose"][:, :, [0]+[ii+1 for ii in c]] for c in context]),
                                    'randomref':  None if not model_kwargs[0].__contains__('randomref') else torch.cat([model_kwargs[0]["randomref"][:, :, c] for c in context]),
                                    }, 
                                    {
                                    'y': None,
                                    "local_image": None, 
                                    'image': None,
                                    'randomref': None,
                                    'dwpose': None, 
                                    }]
            
            if model_kwargs[0].__contains__('pose_embedding'):
                model_kwargs_new[0]['pose_embedding'] = torch.cat([model_kwargs[0]["pose_embedding"][:, :, c] for c in context])

            if guide_scale is None:
                out = model(latent_model_input, self._scale_timesteps(t), **model_kwargs)
                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + out
                    counter[:, :, c] = counter[:, :, c] + 1
            else:
                # classifier-free guidance
                # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
                # assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
                y_out = model(latent_model_input, self._scale_timesteps(t).repeat(bs_context), **model_kwargs_new[0])
                u_out = model(latent_model_input, self._scale_timesteps(t).repeat(bs_context), **model_kwargs_new[1])
                dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
                for j, c in enumerate(context):
                    noise_pred[:, :, c] = noise_pred[:, :, c] + y_out[j:j+1]
                    noise_pred_uncond[:, :, c] = noise_pred_uncond[:, :, c] + u_out[j:j+1]
                    counter[:, :, c] = counter[:, :, c] + 1
                
        noise_pred = noise_pred / counter
        noise_pred_uncond = noise_pred_uncond / counter
        out = torch.cat([
                    noise_pred_uncond[:, :dim] + guide_scale * (noise_pred[:, :dim] - noise_pred_uncond[:, :dim]),
                    noise_pred[:, dim:]], dim=1) # guide_scale=2.5

        
        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)
        
        # compute mean and x0
        if self.mean_type == 'x_{t-1}':
            mu = out  # x_{t-1}
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                 _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
        elif self.mean_type == 'x0':
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'eps':
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == 'v':
            x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        
        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    @torch.no_grad()
    def ddim_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, context_size=32, context_stride=1, context_overlap=4, context_batch_size=1):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps
        
        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale, context_size, context_stride, context_overlap, context_batch_size)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0
    
    @torch.no_grad()
    def ddim_sample_loop(self, noise, context_size, context_stride, context_overlap, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0, context_batch_size=1):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        from tqdm import tqdm
        
        for step in tqdm(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta, context_size=context_size, context_stride=context_stride, context_overlap=context_overlap, context_batch_size=context_batch_size)
        return xt
    
    @torch.no_grad()
    def ddim_reverse_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas_next = _i(
            torch.cat([self.alphas_cumprod, self.alphas_cumprod.new_zeros([1])]),
            (t + stride).clamp(0, self.num_timesteps), xt)
        
        # reverse sample
        mu = torch.sqrt(alphas_next) * x0 + torch.sqrt(1 - alphas_next) * eps
        return mu, x0
    
    @torch.no_grad()
    def ddim_reverse_sample_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        # prepare input
        b = x0.size(0)
        xt = x0

        # reconstruction steps
        steps = torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp, percentile, guide_scale, ddim_timesteps)
        return xt
    
    @torch.no_grad()
    def plms_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
                eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            return eps
        
        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0
        
        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1] + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2] - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    @torch.no_grad()
    def plms_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // plms_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        eps_cache = []
        for step in steps:
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, plms_timesteps, eps_cache)
            
            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
        #return t.float()



def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


def context_scheduler(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = False,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1
    )

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


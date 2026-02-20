import torch
import zuko
class Distr(torch.distributions.Distribution):
    
    def denormalize_samples(self, samples):
        return samples
    
    def normalize_samples(self, samples):
        return samples
    
    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError
    
    def log_prob(self, value):
        raise NotImplementedError
        
    def entropy(self, num_samples=10000):
        # Draw samples from the distribution
        samples = self.rsample((num_samples,))
        
        # Calculate log probabilities for these samples
        log_probs = self.log_prob(samples)
        
        # Entropy is the negative expected log probability
        entropy_estimate = -log_probs.mean()
        return entropy_estimate
    

class UniformDist(Distr):
    def __init__(self, low, high, device):
        self.low = low
        self.high = high
        self.device = device
        self.ndim = len(low)
        diff = self.high - self.low
        self.fixed = diff.abs() < 1e-12  # degenerate dims mask
        self.free = ~self.fixed
        self.uniform = torch.distributions.Uniform(torch.tensor(low), torch.tensor(high))
        # IMPORTANT: Uniform is only over free dims
        if self.free.any():
            self.uniform = torch.distributions.Uniform(self.low[self.free], self.high[self.free])
        else:
            self.uniform = None
    def volume(self):
        """
        Calculate the volume of the hyper-rectangle defined by self.low and self.high.
        """
        diff = self.high - self.low
        diff_safe = torch.where(diff == 0, torch.ones_like(diff), diff)
        return diff_safe.prod()
    def rsample(self, sample_shape=torch.Size()):
        shape = tuple(sample_shape) + (self.ndim,)
        out = self.low.expand(shape).clone()
        if self.uniform is not None:
            out[..., self.free] = self.uniform.rsample(sample_shape)
        return out
    def entropy(self):
        diff = (self.high - self.low).abs()
        fixed = diff < 1e-12
        diff_safe = torch.where(fixed, torch.ones_like(diff), diff)
        return torch.log(diff_safe).masked_fill(fixed, 0.0).sum()

    def log_prob(self, value, tol: float = 1e-6):
        value = value.to(self.device)

        if self.uniform is not None:
            lp = self.uniform.log_prob(value[..., self.free]).sum(-1)
        else:
            lp = torch.zeros(value.shape[:-1], device=value.device, dtype=value.dtype)

        if self.fixed.any():
            ok = (value[..., self.fixed] - self.low[self.fixed]).abs().le(tol).all(dim=-1)
            lp = torch.where(ok, lp, torch.full_like(lp, -torch.inf))

        return lp
    
class MultivariateNormalDist(Distr):
    def __init__(self, mean, cov, low, high):
        self.mv_mean = mean
        self.mv_cov = cov
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.ndim = self.mv_mean.shape[0]
        self.multinorm = torch.distributions.MultivariateNormal(self.mv_mean, self.mv_cov)
        
    
    def entropy(self, **kwargs):
        return self.multinorm.entropy()

    def rsample(self, sample_shape=torch.Size()):
        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        samples = self.multinorm.rsample(sample_shape)
        
        # Check which samples are out of bounds
        valid_mask = torch.all((samples >= self.low) & (samples <= self.high), dim=-1)
        invalid_mask = ~valid_mask
        
        # Generate uniform samples for the invalid ones
        uniform_samples = torch.rand(invalid_mask.sum(), self.ndim, device=samples.device)
        uniform_samples = uniform_samples * (self.high - self.low) + self.low
        
        # Replace invalid samples with uniform samples
        samples[invalid_mask] = uniform_samples
        
        return samples
    
    def log_prob(self, value):
        return self.multinorm.log_prob(value)
    

class NormFlowDist(Distr):
    def __init__(self, low, high, ndim):
        bins = 8
        self.ndim = ndim
        self.device = "cuda"
        self.low = low.to(self.device)
        self.high = high.to(self.device)
        
        self.scale = 10.0
        maf = zuko.flows.MAF(
            features=self.ndim,
            context=0,
            univariate=zuko.transforms.MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            hidden_features=(64, 64),
            transforms=3
        )
        
        self.flow = zuko.flows.Flow(maf.transform.inv, maf.base).to(self.device)

    
    def normalize_samples(self, samples):
        """Normalize samples into standardized space"""

        if samples.ndim == 1:
            samples.reshape(1, -1)

        return ((samples - (self.low+self.high)/2.0  ) / (self.high - self.low)) * self.scale


    def denormalize_samples(self, samples):
        """Denormalize samples back in their true space"""
        samples = samples.to(self.device)
        if samples.ndim == 1:
            samples.reshape(1, -1)

        return (self.high - self.low) * samples / self.scale + (self.low+self.high)/2.0

    def log_prob(self, x):
        norm_x = self.normalize_samples(x.to(self.device))
        return self.flow().log_prob(norm_x.type(torch.FloatTensor).to(self.device))

           
    def get_params(self):
        return self.flow.parameters()
    
    def clone(self):
        # Create a new instance
        new_instance = NormFlowDist(self.low.clone(), self.high.clone(), self.ndim)
        
        # Copy the flow parameters
        new_instance.flow.load_state_dict(self.flow.state_dict())
        
        return new_instance
    
    def rsample(self, sample_shape=torch.Size()):
         
        # Normalized Truncated Normal distribution in [0, 1]
        n_valids = 0
        n_samples = sample_shape[0]
        non_valid_mask = torch.ones((n_samples)).bool()
        samples = torch.zeros((n_samples, self.ndim)).type(torch.FloatTensor).to(self.device)

        n_iters = 0
        while n_valids < n_samples:
            if(n_iters>0):
                print("regenerating iter {}".format(str(n_iters)))        
            norm_samples, _ = self.flow().rsample_and_log_prob((non_valid_mask.int().sum(),))
            norm_samples = norm_samples.type(torch.FloatTensor).to(self.device)
            samples[non_valid_mask] = self.denormalize_samples(norm_samples)
            mask_low = torch.greater_equal(samples, self.low.view(1,-1))
            mask_high = torch.less_equal(samples, self.high.view(1,-1))
            per_sample_mask_with_dim = torch.cat([mask_low, mask_high], dim=-1) # (n_samples, 2*ndim)
            per_sample_mask = torch.all(per_sample_mask_with_dim, dim=-1)
            non_valid_mask = ~per_sample_mask
            n_valids = per_sample_mask.int().sum()
            n_iters += 1
            

        if n_iters >= 10:
            print('WARNING! Sampling through the truncated normal took {n_iters} >= 10 iterations for resampling.')
        
        return samples
    
    
class BetasDist(Distr):
    def __init__(self, alphas, betas, low, high):
        super().__init__(validate_args=False)
        self.alphas = alphas
        self.betas = betas
        self.low = torch.as_tensor(low, device=self.alphas.device, dtype=self.alphas.dtype)
        self.high = torch.as_tensor(high, device=self.alphas.device, dtype=self.alphas.dtype)
        self.dists = [torch.distributions.Beta(a, b) for a, b in zip(self.alphas, self.betas)]
        self.ndim = len(self.dists)

    def rsample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape, device=self.low.device) if isinstance(sample_shape, int) else torch.Size(sample_shape)
        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        samples = torch.stack([d.rsample(sample_shape) for d in self.dists], dim=-1).type(torch.FloatTensor).to(device=self.low.device)
        
        # Transform samples to the desired range
        output = self.low + (self.high - self.low) * samples
        
        # Check which samples are out of bounds
        valid_mask = torch.all((output >= self.low) & (output <= self.high), dim=-1)
        invalid_mask = ~valid_mask
        
        # Generate uniform samples for the invalid ones
        uniform_samples = torch.rand(invalid_mask.sum(), self.ndim, device=samples.device)
        uniform_samples = uniform_samples * (self.high - self.low) + self.low
        
        # Replace invalid samples with uniform samples
        output[invalid_mask] = uniform_samples
        
        return output
    
    def log_prob(self, value):
        # value: (N, D)
        value = value.to(device=self.low.device, dtype=self.low.dtype)

        scale = (self.high - self.low)
        fixed = scale.abs() < 1e-12                          # (D,)
        scale_safe = torch.where(fixed, torch.ones_like(scale), scale)

        # normalize to (0,1)
        eps = 1e-6
        transformed_value = (value - self.low) / scale_safe  # (N, D)
        transformed_value = transformed_value.clamp(eps, 1.0 - eps)

        # per-dim logprob, but ignore fixed dims
        log_probs_per_dim = []
        for i, d in enumerate(self.dists):
            v = transformed_value[:, i]
            lp = d.log_prob(v)                               # (N,)
            if fixed[i]:
                lp = torch.zeros_like(lp)                    # deterministic dim => no contribution
            log_probs_per_dim.append(lp)

        log_probs = torch.stack(log_probs_per_dim, dim=0)    # (D, N)
        return log_probs.sum(dim=0)     

    def to_flat(self):
        """Convert distribution parameters to a flat array."""
        return torch.cat([self.alphas, self.betas])
    def entropy(self):
         # base beta entropy on [0,1]
        base = torch.stack([d.entropy() for d in self.dists]).sum()
        # add scaling term for non-degenerate dims
        diff = (self.high - self.low).abs()
        fixed = diff < 1e-12
        diff_safe = torch.where(fixed, torch.ones_like(diff), diff)
        jac = torch.log(diff_safe).masked_fill(fixed, 0.0).sum()
        return base + jac
    @staticmethod
    def from_flat(x, low, high):
        # x must already be a torch tensor with grad
        assert torch.is_tensor(x)

        # DO NOT do torch.tensor(x) or x.detach()
        # low/high can be tensors already; just ensure device/dtype match x
        low = low.to(device=x.device, dtype=x.dtype)
        high = high.to(device=x.device, dtype=x.dtype)

        # if x is (2D,) or similar, slice WITHOUT breaking graph
        # example if x packs alpha/beta:
        D = low.numel()
        alpha = x[:D]
        beta  = x[D:2*D]

        return BetasDist(alpha, beta, low, high)

    def kl_divergence(self, other):
        """
        Compute KL divergence between this BetasDist and another BetasDist or UniformDist.
        """
        kl_div = self.alphas.new_tensor(0.0)
        eps = 1e-12
        for i in range(self.ndim):
            p_dist = torch.distributions.Beta(self.alphas[i], self.betas[i])
            len_p = (self.high[i] - self.low[i])
            len_p_abs = len_p.abs()

            if isinstance(other, UniformDist):
                len_q = (other.high[i] - other.low[i])
                len_q_abs = len_q.abs()
            elif isinstance(other, BetasDist):
                len_q = (other.high[i] - other.low[i])
                len_q_abs = len_q.abs()
            else:
                raise ValueError(f"Unsupported distribution type: {type(other)}")

            # If either side is degenerate, treat as deterministic => skip
            if (len_p_abs < eps) or (len_q_abs < eps):
                continue
            if isinstance(other, UniformDist):
                # Scale down the uniform distribution to [0, 1]
                q_dist = torch.distributions.Uniform(0, 1)
                # Compute KL divergence
                kl_i = torch.distributions.kl_divergence(p_dist, q_dist)
                # Adjust for the change in scale
                kl_i = kl_i - torch.log(len_p_abs)
            elif isinstance(other, BetasDist):
                q_dist = torch.distributions.Beta(other.alphas[i], other.betas[i])
                kl_i = torch.distributions.kl_divergence(p_dist, q_dist)
                # Adjust for different bounds if necessary
                kl_i = kl_i + torch.log(len_q_abs / len_p_abs)
            else:
                raise ValueError(f"Unsupported distribution type: {type(other)}")
            kl_div += kl_i
        return kl_div

class BoundarySamplingDist(Distr):
    def __init__(self, low, high, boundary_prob=0.5):
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.ndim = len(low)
        self.boundary_prob = boundary_prob
        
    def rsample(self, sample_shape=torch.Size()):
        n_samples = sample_shape[0] if len(sample_shape) > 0 else 1
        samples = torch.zeros((n_samples, self.ndim))
        
        # Decide which samples will be on the boundary
        boundary_mask = torch.rand(n_samples) < self.boundary_prob
        # For non-boundary samples, sample uniformly
        non_boundary_samples = torch.rand((n_samples, self.ndim), device=self.low.device) * (self.high - self.low) + self.low
        
        # For boundary samples, choose a random dimension and boundary
        boundary_dims = torch.randint(0, self.ndim, (n_samples,))
        boundary_is_high = torch.rand(n_samples) < 0.5
        
        for i in range(n_samples):
            if boundary_mask[i]:
                samples[i] = non_boundary_samples[i]
                dim = boundary_dims[i]
                samples[i, dim] = self.high[dim] if boundary_is_high[i] else self.low[dim]
            else:
                samples[i] = non_boundary_samples[i]
        
        return samples
    
    def log_prob(self, value):
        # Check if the value is within the bounds
        in_range = torch.all((value >= self.low) & (value <= self.high), dim=-1)
        
        # Calculate the volume of the sampling space
        diff = self.current_high - self.current_low
        diff_safe = torch.where(diff == 0, torch.ones_like(diff), diff)
        volume =  diff_safe.prod()
        # Calculate the log probability for uniform sampling within the bounds
        log_prob_uniform = -torch.log(volume)
        
        # Calculate the log probability for boundary sampling
        # This is an approximation
        num_dims = len(self.low)
        log_prob_boundary = torch.log(torch.tensor(self.boundary_prob / (2 * num_dims)))
        
        # Combine probabilities
        log_prob = torch.where(
            in_range,
            torch.log(
                (1 - self.boundary_prob) * torch.exp(log_prob_uniform) +
                self.boundary_prob * torch.exp(log_prob_boundary)
            ),
            torch.tensor(float('-inf'))
        )
        
        return log_prob
    

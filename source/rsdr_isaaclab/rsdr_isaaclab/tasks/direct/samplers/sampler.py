import torch
import torch.nn as nn
import torch.distributions as D
from .distributions import UniformDist, BetasDist, BoundarySamplingDist, NormFlowDist, MultivariateNormalDist
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize, Bounds
import time
class LearnableSampler(nn.Module):
    def __init__(self, cfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.num_params = cfg.total_params  # Total scalar parameters
        self.name="default"
        init = []        
        low = []
        high = []
        
        for p in cfg.params:
            # Helper to broadcast scalar to list if size > 1
            def ensure_list(val, size):
                if isinstance(val, (int, float)):
                    return [val] * size
                return val # Assume list/tuple
            
            p_init = ensure_list(p.init_params, p.size) # Low or Mean
            p_low = ensure_list(p.hard_bounds[0], p.size)
            p_high = ensure_list(p.hard_bounds[1], p.size)
            
            init.extend(p_init)
            low.extend(p_low)
            high.extend(p_high)

        # Fixed Constants
        self.init = torch.tensor(init, device=device, dtype=torch.float32)
        self.low = torch.tensor(low, device=device, dtype=torch.float32)
        self.high = torch.tensor(high, device=device, dtype=torch.float32)
        self.mid = (self.low + self.high) / 2.0
        self.current_dist = UniformDist(self.low, self.high, self.device)
        print("Reference sampler with low:", self.low
              , "high:", self.high, "init:", self.init)
    def sample(self, num_samples: int) -> torch.Tensor:
        return self.current_dist.rsample((num_samples,))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
    
        return self.current_dist.log_prob(value)
    def volume(self, low, high):
        diff = high - low
        diff_safe = torch.where(diff == 0, torch.ones_like(diff), diff)
        return diff_safe.prod()
    def update(self, contexts, returns):
        pass
    
    def get_train_dist(self):
        return UniformDist(self.low, self.high, self.device)
    
    def get_test_dist(self):
        return UniformDist(self.low, self.high, self.device)
        
class NoDR(LearnableSampler):
    def __init__(self, cfg, device: str, **kwargs):
        super().__init__(cfg, device)
        self.name = "NoDR"

    def get_train_dist(self):
        return UniformDist(self.init, self.init, self.device)
class UDR(LearnableSampler):
    def __init__(self, cfg, device: str, **kwargs):
        super().__init__(cfg, device)
        self.name = "UDR"
    def get_train_dist(self):
        return UniformDist(self.low, self.high, self.device)
class ADR(LearnableSampler):
    def __init__(self, cfg, device: str,
                 boundary_prob=0.8, 
                 success_threshold=0.8, 
                 expansion_factor=1.1, 
                 initial_dr_percentage=0.2,
                **kwargs):
        super().__init__(cfg, device)
        self.name = "ADR"
        self.ndim = len(self.low)
        self.lower_threshold = success_threshold/2.0
        self.upper_threshold = success_threshold
        self.expansion_factor = expansion_factor
        self.boundary_prob = boundary_prob
        
        mid_range = (torch.tensor(self.low) + torch.tensor(self.high)) / 2
        # interval = (torch.tensor(domain_range.high) - torch.tensor(domain_range.low)) * initial_dr_percentage
        span = self.high - self.low
        half_width = 0.5 * span * initial_dr_percentage
        self.current_low = mid_range - half_width
        self.current_high = mid_range + half_width
        # self.current_dist = BoundarySamplingDist(self.current_low, self.current_high, self.boundary_prob)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.get_train_dist().rsample((num_samples,))
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.get_train_dist().log_prob(value)
    
    def get_train_dist(self):
        return BoundarySamplingDist(self.current_low, self.current_high, self.boundary_prob)

    def update(self, contexts, returns):
        returns = returns.view(-1).to(device=self.device)
        if not torch.is_floating_point(returns):
            returns = returns.float()
         # Randomly select a dimension to update
        dim_t = torch.randint(0, self.ndim, (1,), device=self.device, dtype=torch.long)  # shape (1,)

        # gather the chosen dim column without python int
        ctx_dim = contexts.index_select(1, dim_t).squeeze(1)  # (N,)

        low_b  = self.current_low.index_select(0, dim_t)   # (1,)
        high_b = self.current_high.index_select(0, dim_t)  # (1,)

        atol = torch.tensor(1e-3, device=self.device, dtype=torch.float32)
        rtol = torch.tensor(0.0,  device=self.device, dtype=torch.float32)

        low_mask  = torch.isclose(ctx_dim, low_b,  atol=atol, rtol=rtol)   # (N,)
        high_mask = torch.isclose(ctx_dim, high_b, atol=atol, rtol=rtol)   # (N,)

        # masked mean without branching / sync:
        def masked_mean_sumcount(x, mask):
            m = mask.to(x.dtype)
            count = m.sum()                       # scalar tensor
            s = (x * m).sum()                     # scalar tensor
            mean = s / torch.clamp(count, min=1)  # safe
            mean = mean * (count > 0).to(x.dtype) # zero if empty
            return mean

        low_success_rate  = masked_mean_sumcount(returns, low_mask)   # scalar tensor
        high_success_rate = masked_mean_sumcount(returns, high_mask)  # scalar tensor
        print("Low boundary reward: "+str(low_success_rate))
        print("High boundary reward: "+str(high_success_rate))
        midpoint = 0.5 * (low_b + high_b)  # (1,)
        ef = torch.tensor(float(self.expansion_factor), device=self.device, dtype=torch.float32)

        upper = torch.tensor(float(self.upper_threshold), device=self.device, dtype=torch.float32)
        lower = torch.tensor(float(self.lower_threshold), device=self.device, dtype=torch.float32)

        # ---- update low ----
        low_expand   = low_success_rate  > upper
        low_contract = low_success_rate  < lower

        low_floor = self.low.index_select(0, dim_t)  # (1,)

        new_low_expand   = torch.maximum(midpoint - (midpoint - low_b) * ef, low_floor)
        new_low_contract = torch.minimum(midpoint - (midpoint - low_b) / ef, midpoint)

        new_low = torch.where(low_expand, new_low_expand,
                torch.where(low_contract, new_low_contract, low_b))

        self.current_low.scatter_(0, dim_t, new_low)

        # ---- update high ----
        high_expand   = high_success_rate > upper
        high_contract = high_success_rate < lower

        high_ceil = self.high.index_select(0, dim_t)  # (1,)

        new_high_expand   = torch.minimum(midpoint + (high_b - midpoint) * ef, high_ceil)
        new_high_contract = torch.maximum(midpoint + (high_b - midpoint) / ef, midpoint)

        new_high = torch.where(high_expand, new_high_expand,
                torch.where(high_contract, new_high_contract, high_b))

        self.current_high.scatter_(0, dim_t, new_high)

        print(f"Current domain: Low = {self.current_low}, High = {self.current_high}")
        print("Current domain volume:", self.volume(self.current_low, self.current_high).item())
        print("Reference domain volume:", self.volume(self.low, self.high).item())

class DORAEMON(LearnableSampler):
    def __init__(self, cfg, device: str,
                 success_threshold: float,
                 kl_upper_bound: float = 0.1,
                 init_beta_param: float = 100.,
                 success_rate_condition: float = 0.5,
                 hard_performance_constraint: bool = True,
                 train_until_performance_lb: bool = True,
                 **kwargs):
        device = torch.device("cpu")
        super().__init__(cfg, device)
        self.name = "DORAEMON"
        self.success_threshold = success_threshold
        self.success_rate_condition = success_rate_condition
        self.kl_upper_bound = kl_upper_bound
        self.train_until_performance_lb = train_until_performance_lb
        self.hard_performance_constraint = hard_performance_constraint
        self.train_until_done = False 
        self.ndim = len(self.low)
        
        self.min_bound = 0.8
        self.max_bound = init_beta_param + 10
        
        # Initialize distributions
        self.current_dist = self._create_initial_distribution(init_beta_param)
        self.target_dist = self._create_target_distribution()
        
        self.current_iter = 0
        self.distr_history = []

    
    def _create_initial_distribution(self, init_beta_param):
        return BetasDist(torch.ones(self.ndim, device=self.low.device) * init_beta_param, torch.ones(self.ndim, device=self.low.device) * init_beta_param, self.low, self.high)

    def _create_target_distribution(self):
        return UniformDist(self.low, self.high, self.device)

    def get_train_dist(self):
        return self.current_dist

    def get_test_dist(self):
        return self.target_dist
    def entropy(self):
        return self.current_dist.entropy().sum().item()
    def get_feasible_starting_distr(self, x0_opt, obj_fn, obj_fn_prime, kl_constraint_fn, kl_constraint_fn_prime):
        """
        Solves the inverted problem
        max J(phi_i+1) s.t. KL(phi_i+1 || phi_i) < eps
        to find an initial feasible distribution
        """
        def negative_obj_fn_with_grad(x_opt):
            try:
                obj_val = obj_fn(x_opt)
                obj_grad = obj_fn_prime(x_opt)
                
                # Check for invalid values
                if np.any(np.isnan(obj_val)) or np.any(np.isinf(obj_val)):
                    return np.inf, np.zeros_like(x_opt)
                if np.any(np.isnan(obj_grad)) or np.any(np.isinf(obj_grad)):
                    return obj_val, np.zeros_like(x_opt)
                    
                return -1 * obj_val, -1 * obj_grad
            except Exception as e:
                print(f"Warning: Error in objective function: {e}")
                return np.inf, np.zeros_like(x_opt)

        def safe_kl_constraint_fn(x_opt):
            try:
                val = kl_constraint_fn(x_opt)
                return np.clip(val, -1e10, 1e10)  # Clip to prevent extreme values
            except Exception as e:
                print(f"Warning: Error in KL constraint function: {e}")
                return np.inf

        def safe_kl_constraint_fn_prime(x_opt):
            try:
                grad = kl_constraint_fn_prime(x_opt)
                # Clip gradients to prevent numerical instability
                return np.clip(grad, -1e10, 1e10)
            except Exception as e:
                print(f"Warning: Error in KL constraint gradient: {e}")
                return np.zeros_like(x_opt)

        constraints = []
        constraints.append(
            NonlinearConstraint(
                fun=safe_kl_constraint_fn,
                lb=-np.inf,
                ub=self.kl_upper_bound-1e-5,
                jac=safe_kl_constraint_fn_prime,
                keep_feasible=True,
            )
        )

        # Add bounds to prevent extreme values
        bounds = Bounds(
            lb=-1e3 * np.ones_like(x0_opt),
            ub=1e3 * np.ones_like(x0_opt)
        )

        start = time.time()
        print("Starting optimization 2")
        
        try:
            result = minimize(
                negative_obj_fn_with_grad,
                x0_opt,
                method="trust-constr",
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={
                    "gtol": 1e-4,
                    "xtol": 1e-6,
                    "maxiter": 100,
                    "initial_tr_radius": 1.0,  # Start with a smaller trust region
                    "initial_constr_penalty": 1.0
                }
            )
        except Exception as e:
            print(f"Optimization failed with error: {e}")
            return None, None, False

        print(f"scipy inverted problem optimization time (s): {round(time.time() - start, 2)}")

        if not result.success:
            print(f"Optimization failed with message: {result.message}")
            return None, None, False
        else:
            feasible_x0_opt = result.x
            curr_step_kl = safe_kl_constraint_fn(feasible_x0_opt)
            
            # Verify the result is valid
            if np.any(np.isnan(feasible_x0_opt)) or np.any(np.isinf(feasible_x0_opt)):
                print("Warning: Optimization returned invalid values")
                return None, None, False
                
            return feasible_x0_opt, curr_step_kl, True


    def update(self, contexts, returns):
        self.current_iter += 1

        print("Updating DORAEMON")

        # Convert to numpy and ensure double precision
        contexts = contexts.detach().to(device=self.low.device, dtype=torch.float64)
        returns  = returns.detach().to(device=self.low.device, dtype=torch.float64)

        print("Contexts shape:", contexts.shape)
        print("Returns shape:", returns.shape)
        print("Contexts mean/min/max:", contexts.mean().item(), contexts.min().item(), contexts.max().item())
        print("Returns mean/min/max:", returns.mean().item(), returns.min().item(), returns.max().item())

        """
            2. Optimize KL(phi_i+1 || phi_target) s.t. J(phi_i+1) > performance_bound & KL(phi_i+1 || phi_i) < KL_bound
        """
        constraints = []


        def kl_constraint_fn(x_opt):
            """Compute KL-divergence between current and proposed distribution."""
            x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
            proposed_distr = BetasDist.from_flat(x, self.low, self.high)
            kl_divergence = self.current_dist.kl_divergence(proposed_distr)
            return kl_divergence.detach().cpu().numpy() 

        def kl_constraint_fn_prime(x_opt):
            """Compute the derivative for the KL-divergence (used for scipy optimizer)."""
            with torch.enable_grad():
                x_opt = torch.tensor(x_opt, requires_grad=True)
                x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
                proposed_distr = BetasDist.from_flat(x, self.low, self.high)
                kl_divergence = self.current_dist.kl_divergence(proposed_distr)
                grads = torch.autograd.grad(kl_divergence, x_opt)
                return np.concatenate([g.detach().cpu().numpy() for g in grads])

        constraints.append(
            NonlinearConstraint(
                fun=kl_constraint_fn,
                lb=-np.inf,
                ub=self.kl_upper_bound,
                jac=kl_constraint_fn_prime,
                keep_feasible=self.hard_performance_constraint
            )
        )

        def performance_constraint_fn(x_opt):
            """Compute the expected performance under the proposed distribution."""
            # print("x_opt mean/min/max:", x_opt.mean().item(), x_opt.min().item(), x_opt.max().item())

            x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
            # print("x mean/min/max:", x.mean().item(), x.min().item(), x.max().item())
            proposed_distr = BetasDist.from_flat(x, self.low, self.high)
            
            log_prob_proposed = proposed_distr.log_prob(contexts)
            log_prob_current = self.current_dist.log_prob(contexts)
            
            importance_sampling = torch.exp(log_prob_proposed - log_prob_current)
            # print("log_prob_proposed mean/min/max:", log_prob_proposed.mean().item(), log_prob_proposed.min().item(), log_prob_proposed.max().item())
            # print("log_prob_current mean/min/max:", log_prob_current.mean().item(), log_prob_current.min().item(), log_prob_current.max().item())
            if torch.any(torch.isnan(importance_sampling)) or torch.any(torch.isinf(importance_sampling)):
                print("Warning: NaN or Inf in importance sampling")
                print("log_prob_proposed:", log_prob_proposed)
                print("log_prob_current:", log_prob_current)
                importance_sampling = torch.nan_to_num(importance_sampling, nan=1.0, posinf=1.0, neginf=1.0)
            
            perf_values = torch.tensor(returns.detach() >= self.success_threshold, dtype=torch.float64)
            performance = torch.mean(importance_sampling * perf_values)
            
            if torch.isnan(performance) or torch.isinf(performance):
                print("Warning: NaN or Inf in performance")
                performance = torch.tensor(0.0, dtype=torch.float64)
            
            return performance.detach().cpu().numpy()

        def performance_constraint_fn_prime(x_opt):
            """Compute the derivative for the performance-constraint (used for scipy optimizer)."""
            with torch.enable_grad():
                x_opt = torch.tensor(x_opt, requires_grad=True)
                x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
                proposed_distr = BetasDist.from_flat(x, self.low, self.high)
                
                log_prob_proposed = proposed_distr.log_prob(contexts)
                log_prob_current = self.current_dist.log_prob(contexts)
                
                importance_sampling = torch.exp(log_prob_proposed - log_prob_current)
                
                if torch.any(torch.isnan(importance_sampling)) or torch.any(torch.isinf(importance_sampling)):
                    print("Warning: NaN or Inf in importance sampling (prime)")
                    importance_sampling = torch.nan_to_num(importance_sampling, nan=1.0, posinf=1.0, neginf=1.0)
                
                perf_values = torch.tensor(returns.detach() >= self.success_threshold, dtype=torch.float64)
                performance = torch.mean(importance_sampling * perf_values)
                
                if torch.isnan(performance) or torch.isinf(performance):
                    print("Warning: NaN or Inf in performance (prime)")
                    return np.zeros_like(x_opt.detach().cpu().numpy())

                grads = torch.autograd.grad(performance, x_opt)
                grad_np = np.concatenate([g.detach().cpu().numpy() for g in grads])
                
                if np.any(np.isnan(grad_np)) or np.any(np.isinf(grad_np)):
                    print("Warning: NaN or Inf in gradients")
                    grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
                
                return grad_np

        constraints.append(
            NonlinearConstraint(
                fun=performance_constraint_fn,
                lb=self.success_rate_condition-1e-4,  # scipy would still complain if x0 is very close to the boundary
                ub=np.inf,
                jac=performance_constraint_fn_prime,
                keep_feasible=self.hard_performance_constraint
            )
        )



        def objective_fn(x_opt):
            """Minimize KL-divergence between the current and the target distribution,
                s.t. previously defined constraints."""
            with torch.enable_grad():
                x_opt = torch.tensor(x_opt, requires_grad=True, dtype=torch.float64)
                x = self._sigmoid(x_opt, self.min_bound, self.max_bound)
                proposed_distr = BetasDist.from_flat(x, self.low, self.high)
                kl_divergence = proposed_distr.kl_divergence(self.target_dist)
                
                if not torch.isfinite(kl_divergence):
                    print(f"Warning: Non-finite KL divergence detected: {kl_divergence}")
                    return float('inf'), np.zeros_like(x_opt.detach().cpu().numpy())
                
                grads = torch.autograd.grad(kl_divergence, x_opt, create_graph=True)
                grad_np = np.concatenate([g.detach().cpu().numpy() for g in grads])
                
                if not np.isfinite(grad_np).all():
                    print(f"Warning: Non-finite gradient detected: {grad_np}")
                    return float('inf'), np.zeros_like(x_opt.detach().cpu().numpy())
                return kl_divergence.detach().cpu().numpy(), grad_np


        x0 = self.current_dist.to_flat()
        x0_opt = self._inv_sigmoid(x0, self.min_bound, self.max_bound)
        
        """
            Skip DORAEMON optimization at the beginning until performance_lower_bound is reached 
        """
        if self.train_until_performance_lb and not self.train_until_done:
            if performance_constraint_fn(x0_opt) < self.success_rate_condition:
                # Skip DORAEMON update
                print(f'--- DORAEMON iter {self.current_iter} skipped as performance lower bound has not been reached yet. Mean reward {performance_constraint_fn(x0_opt)} < {self.success_rate_condition}')
                return
            else:
                # Skip iterations only once, until you reach it the first time
                self.train_until_done = True
                self.n_iter_skipped = 0


        """
            Start from a feasible distribution within the trust region Kl(p||.) < eps
        """
        if performance_constraint_fn(x0_opt) < self.success_rate_condition:
            # Performance constraint not satisfied. Find a different initial distribution within the current trust region
            max_perf_x0_opt, curr_step_kl, success = self.get_feasible_starting_distr(x0_opt=x0_opt,
                                                                                        obj_fn=performance_constraint_fn,
                                                                                        obj_fn_prime=performance_constraint_fn_prime,
                                                                                        kl_constraint_fn=kl_constraint_fn,
                                                                                        kl_constraint_fn_prime=kl_constraint_fn_prime)
            if success:
                if performance_constraint_fn(max_perf_x0_opt) >= self.success_rate_condition:
                    # Feasible distribution found, Go on with this new starting distribution
                    x0_opt = max_perf_x0_opt
                    x0 = self._sigmoid(x0_opt, self.min_bound, self.max_bound)

                else:
                    # No feasible distribution within the trust region has been found
                    # Keep training with the max performance distribution within the trust region
                    new_x = self._sigmoid(max_perf_x0_opt, self.min_bound, self.max_bound)
                    self.current_dist = BetasDist.from_flat(new_x, self.low, self.high)
                    print(f'No distribution within the trust region satisfies the performance_constraint. ' \
                            f'Keep training with the max performant distribution in the trust region: {new_x.detach().numpy()} ' \
                            f'Est reward constraint value: {performance_constraint_fn(max_perf_x0_opt)} < {self.success_rate_condition}')
                    return

            else:
                # Inverse opt. problem had an unexpected error
                print('Warning! inverted optimization problem NOT successful.')
        
        print("Starting optimization...")
        
        try:
            result = minimize(
                objective_fn,
                x0_opt,
                method="trust-constr",
                jac=True,
                constraints=constraints,
                options={"gtol": 1e-4, "xtol": 1e-6, 'maxiter': 100},
            )

            print(f"Optimization result: {result}")
            new_x_opt = result.x
            
            # Check validity of new optimum found
            if not result.success:
                print('Warning! optimization NOT successful.')
                
                # If optimization process was not a success
                old_f = objective_fn(x0_opt)[0]
                constraints_satisfied = [const.lb <= const.fun(result.x) <= const.ub for const in constraints]

                if not (all(constraints_satisfied) and result.fun < old_f):  # keep old parameters if update was unsuccessful
                    print(f"Warning! Update effectively unsuccessful, keeping old values parameters.")
                    new_x_opt = x0_opt

            new_x = self._sigmoid(new_x_opt, self.min_bound, self.max_bound)
            self.current_dist = BetasDist.from_flat(new_x, self.low, self.high)
            print(f"New distribution parameters: {new_x}")
            

        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            print("Keeping current distribution")
    
        print("DORAEMON current entropy", self.entropy())
        print("Reference distribution entropy", self.target_dist.entropy().sum().item())
    def _sigmoid(self, x, lb=0.0, up=1.0):
        # Preserve autograd if x is already a tensor
        if torch.is_tensor(x):
            xt = x
        else:
            xt = torch.as_tensor(x, dtype=torch.float64, device=self.low.device)

        lb = torch.as_tensor(lb, dtype=xt.dtype, device=xt.device)
        up = torch.as_tensor(up, dtype=xt.dtype, device=xt.device)
        return (up - lb) / (1 + torch.exp(-xt)) + lb


    def _inv_sigmoid(self, x, lb=0.0, up=1.0):
        if torch.is_tensor(x):
            xt = x
        else:
            xt = torch.as_tensor(x, dtype=torch.float64, device=self.low.device)

        lb = torch.as_tensor(lb, dtype=xt.dtype, device=xt.device)
        up = torch.as_tensor(up, dtype=xt.dtype, device=xt.device)

        eps = 1e-12
        xt = xt.clamp(lb + eps, up - eps)
        return -torch.log((up - lb) / (xt - lb) - 1)

        
class GOFLOW(LearnableSampler):
    def __init__(self,  cfg, device: str,
                 num_training_iters=None, alpha=None, beta=None, max_loss=1e6, **kwargs):
        super().__init__(cfg, device)
        self.name = "GOFLOW"
        self.alpha = alpha  # Weight for entropy maximization (KL to target)
        self.beta = beta    # Weight for similarity constraint (KL to previous)
        self.current_dist = NormFlowDist(
            torch.tensor(self.low),
            torch.tensor(self.high),
            ndim=len(self.low)
        )
        self.dist_optimizer = torch.optim.Adam(self.current_dist.get_params(), lr=1e-3)
        self.dist_optimizer.zero_grad()
        self.num_training_iters = num_training_iters
        self.dist_history = []
        self.target_dist = UniformDist(self.low, self.high, self.device)
        self.max_loss = max_loss  # Add a maximum loss threshold
    
    def get_test_dist(self):
        return self.target_dist

    def get_train_dist(self):
        return self.current_dist

    def update(self, contexts, returns, entropy_update=True):
        print("Updating the GOFLOW distribution")
        R = torch.FloatTensor(returns).flatten().to(self.current_dist.device)
        R_ = (R - R.mean()) / (R.std() + 1e-8)

        previous_dist = self.current_dist.clone()

        for iter in range(self.num_training_iters):
            self.dist_optimizer.zero_grad()

            log_prob = self.current_dist.log_prob(contexts)
            log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)  # Clamp log probabilities

            z_target = self.target_dist.rsample([10000]).to(self.current_dist.device)
            log_p_current = self.current_dist.log_prob(z_target)
            log_p_target = self.target_dist.log_prob(z_target).to(self.current_dist.device)
            
            log_p_current = torch.clamp(log_p_current, min=-1e6, max=1e6)
            log_p_target = torch.clamp(log_p_target, min=-1e6, max=1e6)
            
            if(entropy_update):
                kl_loss_target = self.target_dist.volume()*torch.mean(torch.exp(log_p_current)*log_p_current)
            else:
                kl_loss_target = torch.mean(log_p_target - log_p_current)
            
            with torch.no_grad():
                z_previous = previous_dist.rsample([10000]).to(self.current_dist.device)
                log_p_previous = previous_dist.log_prob(z_previous)
                log_p_previous = torch.clamp(log_p_previous, min=-1e6, max=1e6)
            
            log_p_current_on_previous = self.current_dist.log_prob(z_previous)
            log_p_current_on_previous = torch.clamp(log_p_current_on_previous, min=-1e6, max=1e6)
            
            kl_loss_similarity = torch.mean(log_p_previous - log_p_current_on_previous)

            if(entropy_update):
                reward_loss = self.target_dist.volume()*((R_.detach() * log_prob * torch.exp(log_prob)).mean())
            else:
                reward_loss = -((R_.detach() * log_prob).mean())
            entropy_loss = self.alpha * kl_loss_target
            similarity_loss = self.beta * kl_loss_similarity
            total_loss = reward_loss + entropy_loss + similarity_loss
        
            # Check if loss is finite
            if not torch.isfinite(total_loss):
                print(f"Warning: Non-finite loss detected in iteration {iter}. Skipping update.")
                continue

            # Clip the total loss
            total_loss = torch.clamp(total_loss, max=self.max_loss)

            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.current_dist.get_params(), max_norm=1.0)
            
            self.dist_optimizer.step()
            
            print(f"Iteration {iter}:")
            print(f"  Reward Loss: {reward_loss.item():.4f}")
            print(f"  Entropy Loss (KL to Target): {entropy_loss.item():.4f}")
            print(f"  Similarity Loss (KL to Previous): {similarity_loss.item():.4f}")
            print(f"  Total Loss: {total_loss.item():.4f}")
        
        # Check if the final distribution is valid
        if not self.is_distribution_valid():
            print("Warning: Final distribution is invalid. Reverting to previous distribution.")
            self.current_dist = previous_dist

    def is_distribution_valid(self):
        # Implement checks to ensure the distribution is valid
        # For example, check if the parameters are finite and within expected ranges
        for param in self.current_dist.get_params():
            if not torch.isfinite(param).all():
                return False
        return True

import torch
from torch.optim import Optimizer

class AdaSecant(Optimizer):
    def __init__(self, params,  decay=0.95, gamma_clip=None, damping=1e-7, start_var_reduction=0, delta_clip=None,
                 use_adagrad=True, lower_bound_tau=1.5, upper_bound_tau=1e8, epsilon=1e-6):
        if not 0.0 < decay < 1:
            raise ValueError(f"Invalid decay rate: {decay}. Needs to be between 0 and 1")
        
        self.damping = damping
        self.slow_constant = 2.1
        self.start_var_reduction = start_var_reduction
        self.delta_clip = delta_clip
        self.epsilon = epsilon
        
        defaults = dict(decay=decay, gamma_clip=gamma_clip, damping=damping,
                        use_adagrad=use_adagrad, lower_bound_tau=lower_bound_tau, 
                        upper_bound_tau=upper_bound_tau)
        super(AdaSecant, self).__init__(params, defaults)

    def _init_state(self, state, p):
        """ Initialize the state for each parameter """
        eps = self.damping
        state['step'] = 0
        state['mean_grad'] = torch.zeros_like(p.data) + eps

        if self.defaults['use_adagrad']:
            state['sum_square_grad'] = torch.zeros_like(p.data)

        state['taus'] = torch.ones_like(p.data) * (1+eps) * self.slow_constant
        
        # Numerator of the gamma:
        state['gamma_nume_sqr'] = torch.zeros_like(p.data) + eps
        # Denominator of the gamma:
        state['gamma_deno_sqr'] = torch.zeros_like(p.data) + eps
        # mean_squared_grad := E[g^2]_{t-1}
        state['mean_square_grad'] = torch.zeros_like(p.data) + eps
        # For the covariance parameter := E[\gamma \alpha]_{t-1}
        state['cov_num_t'] = torch.zeros_like(p.data) + eps
        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        state['mean_square_dx'] = torch.zeros_like(p.data) + eps
        
        state['old_plain_grad'] = torch.zeros_like(p.data) + eps
        state['old_grad'] = torch.zeros_like(p.data) + eps

        # The uncorrected gradient of previous of the previous update:
        state['mean_curvature'] = torch.zeros_like(p.data) + eps
        state['mean_curvature_sqr'] = torch.zeros_like(p.data) + eps
        # Initialize the E[\Delta]_{t-1]
        state['mean_dx'] = torch.zeros_like(p.data)
        
        # Block-wise normalize the gradient:
        state['norm_grad'] = torch.zeros_like(p.data) + eps


        
    def _normalize_gradients(self, gradients):
        """ Normalize gradients individually (block normalization) """
        eps = self.damping
        return {p: g / (g.norm(2) + eps) for p, g in gradients.items()}

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            params = group['params']
            gradients = {p: p.grad.data for p in params if p.grad is not None}
            normalized_gradients = self._normalize_gradients(gradients)

            for p in params:
                if p.grad is not None:
                    p.grad.data.copy_(normalized_gradients[p])
                self._update_param(group, p)

    def _update_param(self, group, p):
        """Update parameters and state associated with each parameter."""
        eps = self.damping
        state = self.state[p]
        if not state:
            self._init_state(state, p)

        # Increment step counter
        step = state["step"]
        state["step"] += 1

        # Extract state variables
        mean_grad = state["mean_grad"]
        norm_grad = p.grad.data
        if torch.isnan(norm_grad).any():
            return

        old_plain_grad = state['old_plain_grad']
        old_grad = state["old_grad"]
        cov_num_t = state["cov_num_t"]
        mean_square_dx = state["mean_square_dx"]
        sum_square_grad = state.get('sum_square_grad', None)
        mean_dx = state["mean_dx"]
        mean_square_grad = state["mean_square_grad"]
        taus = state["taus"]
        mean_curvature = state['mean_curvature']
        mean_curvature_sqr = state['mean_curvature_sqr']

        # For the first time-step, assume that delta_x_t := norm_grad
        if step == 0:
            msdx = norm_grad**2
            mdx = norm_grad
        else:
            msdx = mean_square_dx
            mdx = mean_dx


        """ 
        Compute the new updated values
        """
        # print(msdx.abs().min(), mdx.abs().min(), mdx.norm(2), norm_grad.mean(), norm_grad.std(), norm_grad.abs().max(), norm_grad.abs().min())    

        # E[g_i^2]_t    
        new_mean_squared_grad = (mean_square_grad * group['decay'] + torch.square(norm_grad) * (1 - group['decay']))
        # E[g_i]_t
        new_mean_grad = (mean_grad * group['decay'] + norm_grad * (1 - group['decay']))

        mg = new_mean_grad
        mgsq = new_mean_squared_grad

        # Calculate gamma
        new_gamma_nume_sqr = (state['gamma_nume_sqr'] * (1 - 1 / taus) + torch.square((norm_grad - old_grad) * (norm_grad - mg)) / taus)
        new_gamma_deno_sqr = (state['gamma_deno_sqr'] * (1 - 1 / taus) + torch.square((norm_grad-mg) * (old_grad - mg)) / taus)

        gamma = torch.sqrt(new_gamma_nume_sqr + eps) / (torch.sqrt(new_gamma_deno_sqr + eps) + eps)
        if torch.isnan(gamma).any() or torch.isinf(gamma).any():
            print("Gamma has NaN or Inf values.")
            return
        
        # Apply gamma clipping if specified
        if group['gamma_clip'] is not None:
            gamma = torch.clamp(gamma, max=group['gamma_clip'])

        momentum_step = gamma * mg
        corrected_grad_cand = (norm_grad + momentum_step) / (1 + gamma)
        if torch.isnan(corrected_grad_cand).any():
            print("Corrected gradient candidate has NaN values.")
            return

        # For starting the variance reduction.
        if self.start_var_reduction > -1:
            cond = torch.tensor(step >= self.start_var_reduction, dtype=torch.bool)
            corrected_grad = torch.where(cond, corrected_grad_cand, norm_grad)
        else:
            corrected_grad = norm_grad

        if group['use_adagrad']:
            if sum_square_grad is None:
                raise ValueError("sum_square_grad should be initialized but found None.")

            # Update sum of squared gradients
            new_sum_squared_grad = sum_square_grad + torch.square(corrected_grad)
            
            # Compute the root mean square of gradients
            rms_g_t = torch.sqrt(new_sum_squared_grad + eps)
            
            # Ensure rms_g_t is not too small
            rms_g_t = torch.maximum(rms_g_t, torch.tensor(1.0, device=rms_g_t.device))
            
            # Check for NaNs or Infs
            if torch.isnan(rms_g_t).any() or torch.isinf(rms_g_t).any():
                print("RMS gradient has NaN or Inf values.")
                return

        # Use the gradients from the previous update to compute the \nabla f(x_t) - \nabla f(x_{t-1})
        cur_curvature = norm_grad - old_plain_grad
        cur_curvature_sqr = torch.square(cur_curvature)

        # Average curvature
        new_curvature_ave = (mean_curvature * (1 - 1 / taus) + cur_curvature / taus)
        new_curvature_sqr_ave = (mean_curvature_sqr * (1 - 1 / taus) + cur_curvature_sqr / taus)

        epsilon = self.epsilon
        rms_dx_tm1 = torch.sqrt(msdx + epsilon)
        rms_curve_t = torch.sqrt(new_curvature_sqr_ave + epsilon)

        # Update step definition
        delta_x_t = -(rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + epsilon))
        if torch.isnan(delta_x_t).any() or torch.isinf(delta_x_t).any():
            return

        if self.delta_clip:
            delta_x_t = torch.clamp(delta_x_t, -self.delta_clip, self.delta_clip)

        if group['use_adagrad']:
            delta_x_t = delta_x_t * corrected_grad / rms_g_t
        else:
            delta_x_t = delta_x_t * corrected_grad

        new_taus_t = (1 - torch.square(mdx) / (msdx + eps)) * taus + 1.0

        #To compute the E[\Delta^2]_t
        new_mean_square_dx = (msdx * (1 - 1 / taus) + torch.square(delta_x_t) / taus)
        #To compute the E[\Delta]_t
        new_mean_dx = (mdx * (1 - 1 / taus) + delta_x_t / taus)

        outlier_condition = (torch.abs(norm_grad - mg) > (2 * torch.sqrt(mgsq - torch.square(mg)))) | (torch.abs(cur_curvature - new_curvature_ave) > (2 * torch.sqrt(new_curvature_sqr_ave - torch.square(new_curvature_ave))))
        outlier_condition_tensor = outlier_condition.to(p.device)  # Ensure condition is a tensor
        new_taus_t = torch.where(outlier_condition_tensor, torch.tensor(2.2, device=p.device), new_taus_t)

        new_taus_t = torch.clamp(new_taus_t, min=group['lower_bound_tau'], max=group['upper_bound_tau'])
        new_cov_num_t = (cov_num_t * (1 - 1 / taus) + delta_x_t * cur_curvature / taus)

        # Update the state
        state.update({
            "mean_grad": new_mean_grad,
            "mean_square_grad": new_mean_squared_grad,
            "gamma_nume_sqr": new_gamma_nume_sqr,
            "gamma_deno_sqr": new_gamma_deno_sqr,
            "taus": new_taus_t,
            "mean_square_dx": new_mean_square_dx,
            "mean_dx": new_mean_dx,
            "cov_num_t": new_cov_num_t,
            "mean_curvature": new_curvature_ave,
            "mean_curvature_sqr": new_curvature_sqr_ave,
            "old_plain_grad": norm_grad,
        })

        if group['use_adagrad']:
            state["sum_square_grad"] = new_sum_squared_grad

        # Apply the update to the parameter
        p.data.add_(delta_x_t)
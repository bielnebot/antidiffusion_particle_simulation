import torch
from create_dataset import sample_batch

BETA_MIN = 0.00001
BETA_MAX = 0.15 # 0.3


class DiffusionProbabilisticModel():
    def __init__(self, model, n_steps=40):
        super().__init__()

        self.model = model

        # Shape betas as a sigmoid with MIN and MAX and shifted to the side
        betas = torch.linspace(-8, 5, n_steps)
        self.beta = torch.sigmoid(betas) * (BETA_MAX - BETA_MIN) + BETA_MIN

        # self.beta = torch.linspace(BETA_MIN, BETA_MAX, n_steps)
        # self.beta = 0.1 * torch.ones(n_steps)

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps

    def diffuse_forward(self, x0, t):

        t = t - 1  # Start indexing at 0
        beta_forward = self.beta[t]
        alpha_forward = self.alpha[t]
        alpha_cum_forward = self.alpha_bar[t]
        # q(x_t|x_o) = N(sqrt(a_bar) * x_0, (1-a_bar)*I)
        xt = x0 * torch.sqrt(alpha_cum_forward) + torch.randn_like(x0) * torch.sqrt(1. - alpha_cum_forward)
        mu1_scl = torch.sqrt(alpha_cum_forward / alpha_forward)
        mu2_scl = 1. / torch.sqrt(alpha_forward)
        cov1 = 1. - alpha_cum_forward / alpha_forward
        cov2 = beta_forward / alpha_forward
        lam = 1. / cov1 + 1. / cov2
        mu = (x0 * mu1_scl / cov1 + xt * mu2_scl / cov2) / lam
        sigma = torch.sqrt(1. / lam)
        return mu, sigma, xt

    def diffuse_backwards(self, xt, t):
        """
        p(x_t-1|xt,t) = N(x_t-1,mu_theta,cov_theta)
        :param xt:
        :param t:
        :return:
        """

        t = t - 1  # Start indexing at 0
        if t == 0: return None, None, xt
        mu, h = self.model(xt, t).chunk(2, dim=1)
        sigma = torch.sqrt(torch.exp(h))
        # x_t-1 ~ N(mu, sigma)
        samples = mu + torch.randn_like(xt) * sigma
        return mu, sigma, samples

    def denoise_sample(self, size):
        # Sample random noise
        xT = torch.randn((size, 2)) # xT ~ N(0,1)
        # Save all time steps in the list
        samples = [xT]
        for t in range(self.n_steps,0,-1): # t = T, T-1, T-2, ..., 2, 1
            xt = samples[-1]
            _, _, x_t_minus_one = self.diffuse_backwards(xt, t)
            samples.append(x_t_minus_one)
        return samples

if __name__ == "__main__":
    model = DiffusionProbabilisticModel(model=None,n_steps=40)
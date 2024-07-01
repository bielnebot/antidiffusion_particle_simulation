import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_moons


# def sample_batch(size):
#     x, _ = make_swiss_roll(size)
#     x = x[:, [2, 0]] / 10.0 * np.array([1, -1])
#     return 10 * torch.from_numpy(x).float()

# def sample_batch(size):
#     x, _ = make_moons(size)
#     x = x / 10.0 * np.array([1, -1])
#     return 10 * torch.from_numpy(x).float()

# def sample_batch(size):
#     distr = torch.distributions.MultivariateNormal(loc=torch.zeros(2),
#                                                    covariance_matrix=(0.01 ** 2) * torch.eye(2))
#     return distr.sample((1,size))[0]

# def sample_batch(size):
#     distr = torch.distributions.MultivariateNormal(loc=torch.zeros(2),
#                                                    covariance_matrix=(1) * torch.eye(2))
#     return distr.sample((1,size))[0]

def sample_batch(size):
    distr_1 = torch.distributions.MultivariateNormal(loc=torch.tensor([1.0,1.0]),
                                                     covariance_matrix=(0.1 ** 2) * torch.eye(2))
    distr_2 = torch.distributions.MultivariateNormal(loc=torch.tensor([-0.7, -0.3]),
                                                     covariance_matrix=(0.1 ** 2) * torch.eye(2))
    distr_3 = torch.distributions.MultivariateNormal(loc=torch.tensor([-0.35, -1.5]),
                                                     covariance_matrix=(0.1 ** 2) * torch.eye(2))
    return torch.concatenate((distr_1.sample((1, size))[0],
                              distr_2.sample((1, size))[0],
                              distr_3.sample((1, size))[0]),
                             dim=0)


def generate_dataset(n_points=100):
    radius = torch.linspace(start=0.5,end=1.5,steps=n_points)
    theta = torch.linspace(start=0,end=3*torch.pi,steps=n_points)

    dataset = torch.zeros(n_points,2)
    dataset[:, 0] = radius * torch.cos(theta)
    dataset[:, 1] = radius * torch.sin(theta)

    return dataset


if __name__ == "__main__":
    # dataset = generate_dataset()
    dataset = sample_batch(100)
    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.2)
    plt.axis("equal")
    plt.show()

    # a,_ = make_swiss_roll(10)
    # print(a)
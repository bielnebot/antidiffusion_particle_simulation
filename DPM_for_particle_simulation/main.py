import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from create_dataset import sample_batch
from networks import ModelNN
from diffusion_model import DiffusionProbabilisticModel
from plots import plot_forward_and_backward, animate_antidiffusion, animate_antidiffusion_with_trail, \
    plot_trail, plot_beta_comparison, plot_forward_and_backward_with_colors, animate_antidiffusion_with_color

# Training
EPOCHS = 200_000
BATCH_SIZE = 21333 #64_000
LEARNING_RATE = 2e-4

# Diffusion model
T = 40


def train():
    # Set up tensorboard
    writer = SummaryWriter()
    # Initialize model and trainable kernel
    kernel = ModelNN(n=T)
    optimizer = torch.optim.Adam(kernel.parameters(), lr=LEARNING_RATE)
    model = DiffusionProbabilisticModel(kernel, n_steps=T)

    for epoch_i in tqdm(range(EPOCHS)):
        # Sample batch dataset from dataset distribution
        x0 = sample_batch(BATCH_SIZE)
        # Sample t
        t = np.random.randint(2, T + 1)

        # Simulate forward: q(x_t|x_0,t)
        mu_forward, sigma_forward, xt = model.diffuse_forward(x0,t)
        # Simulate backwards: p(x_t-1|xt,t)
        mu_backwards, sigma_backwards, _ = model.diffuse_backwards(xt,t)

        # Compute loss: KL divergence
        loss = torch.log(sigma_backwards/sigma_forward) + \
               (sigma_forward ** 2 + (mu_forward - mu_backwards) ** 2) / (2 * sigma_backwards ** 2) \
               - 0.5
        loss = loss.mean()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss",loss,epoch_i)

        if epoch_i in [2_000, 5_000, 7_000, 10_000, 12_000, 15_000, 17_000, 50_000, 75_000, 100_000, 125_000, 150_000, 175_000]:
            torch.save(kernel.state_dict(), f"./checkpoints/model_3_points_epoch_{epoch_i}.pt")
    # Save the model
    torch.save(kernel.state_dict(), "./checkpoints/model_3_points.pt")


def test():

    # Initialize model and load kernel
    checkpoint_name = "./checkpoints/model_3_points_epoch_100000.pt"
    kernel = ModelNN(n=T)
    kernel.load_state_dict(torch.load(checkpoint_name))
    model = DiffusionProbabilisticModel(kernel,n_steps=T)

    n_datapoints = 5000

    # plot_forward_and_backward(n_datapoints,model,T)

    # plot_forward_and_backward_with_colors(n_datapoints, model, T)

    # animate_antidiffusion_with_color(n_datapoints,model,T)

    # plot_trail(2, model)

    # animate_antidiffusion(n_datapoints,model,T)

    # animate_antidiffusion_with_trail(n_datapoints,model,T)

    plot_beta_comparison(n_datapoints,model,T)

if __name__ == "__main__":
    # train()
    test()

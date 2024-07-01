import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import animation

from create_dataset import sample_batch


def plot_forward_and_backward(n_datapoints,model,T):
    # Plot
    plt.figure(figsize=(10, 4))

    # DIFFUSION
    # Sample batch dataset from dataset distribution: x0 ~ q(x0)
    x0 = sample_batch(n_datapoints)
    _, _, x10 = model.diffuse_forward(x0=x0, t=int(T / 4))
    _, _, x20 = model.diffuse_forward(x0=x0, t=int(T / 2))
    _, _, x30 = model.diffuse_forward(x0=x0, t=int(3 * T / 4))
    _, _, x40 = model.diffuse_forward(x0=x0, t=int(T))
    print(type(x40),x40.shape)
    print(torch.std(x40,dim=0))
    data = [x0, x10, x20, x30, x40]

    for i, t in enumerate([0, int(T/4), int(T), int(3*T/4), int(T) - 1]):
        plt.subplot(2, 5, 1 + i)
        plt.scatter(data[i][:, 0], data[i][:, 1], alpha=.1, s=1,c="k")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i == 0: plt.title(r'$t=0$', fontsize=17)
        if i == 1: plt.title(r'$t=\frac{T}{4}$', fontsize=17)
        if i == 2: plt.title(r'$t=\frac{T}{2}$', fontsize=17)
        if i == 3: plt.title(r'$t=\frac{3T}{4}$', fontsize=17)
        if i == 4: plt.title(r'$t=T$', fontsize=17)
        if i != 0:
            plt.axis("off")

    # ANTI-DIFFUSION
    # Sample batch dataset from noise distribution: xT ~ N(0,1) and simulate backwards
    sample_evolution = model.denoise_sample(n_datapoints)
    for i, t in enumerate([0, int(T/4), int(T/2), int(3*T/4), int(T) - 1]):
        plt.subplot(2, 5, 6 + i)
        current_sample = sample_evolution[int(T) - t].detach().numpy()
        plt.scatter(current_sample[:, 0], current_sample[:, 1], alpha=.1, s=1, c='k')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i != 0:
            plt.axis("off")
    # plt.savefig(f"plots/diffusion_model.png", bbox_inches='tight',dpi=200)
    # plt.close()
    plt.show()


def plot_forward_and_backward_with_colors(n_datapoints,model,T):
    # Plot
    plt.figure(figsize=(10, 4))

    # DIFFUSION
    # Sample batch dataset from dataset distribution: x0 ~ q(x0)
    x0 = sample_batch(n_datapoints)
    _, _, x10 = model.diffuse_forward(x0=x0, t=int(T / 4))
    _, _, x20 = model.diffuse_forward(x0=x0, t=int(T / 2))
    _, _, x30 = model.diffuse_forward(x0=x0, t=int(3 * T / 4))
    _, _, x40 = model.diffuse_forward(x0=x0, t=int(T))
    print(type(x40),x40.shape)
    print(torch.std(x40,dim=0))
    data = [x0, x10, x20, x30, x40]

    for i, t in enumerate([0, int(T/4), int(T), int(3*T/4), int(T) - 1]):
        plt.subplot(2, 5, 1 + i)
        plt.scatter(data[i][:n_datapoints, 0], data[i][:n_datapoints, 1], alpha=.1, s=1,c="g")
        plt.scatter(data[i][n_datapoints:2*n_datapoints, 0], data[i][n_datapoints:2*n_datapoints, 1], alpha=.1, s=1,c="b")
        plt.scatter(data[i][2*n_datapoints:3*n_datapoints, 0], data[i][2*n_datapoints:3*n_datapoints, 1], alpha=.1, s=1,c="r")
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i == 0: plt.title(r'$t=0$', fontsize=17)
        if i == 1: plt.title(r'$t=\frac{T}{4}$', fontsize=17)
        if i == 2: plt.title(r'$t=\frac{T}{2}$', fontsize=17)
        if i == 3: plt.title(r'$t=\frac{3T}{4}$', fontsize=17)
        if i == 4: plt.title(r'$t=T$', fontsize=17)
        if i != 0:
            plt.axis("off")

    # ANTI-DIFFUSION
    # Sample batch dataset from noise distribution: xT ~ N(0,1) and simulate backwards
    sample_evolution = model.denoise_sample(3*n_datapoints)
    # Identify points belonging to each origin
    x_0 = sample_evolution[-1]
    point_1_index = (x_0[:,0] > 0.2).nonzero().view(-1).detach().numpy()
    point_3_index = (x_0[:, 1] < -1).nonzero().view(-1).detach().numpy()
    point_2_index = np.setdiff1d(np.arange(n_datapoints), point_1_index)
    point_2_index = np.setdiff1d(point_2_index, point_3_index)

    for i, t in enumerate([0, int(T/4), int(T/2), int(3*T/4), int(T) - 1]):
        plt.subplot(2, 5, 6 + i)
        current_sample = sample_evolution[int(T) - t].detach().numpy()
        plt.scatter(current_sample[point_1_index, 0], current_sample[point_1_index, 1], alpha=.1, s=1, c='g')
        plt.scatter(current_sample[point_2_index, 0], current_sample[point_2_index, 1], alpha=.1, s=1, c='b')
        plt.scatter(current_sample[point_3_index, 0], current_sample[point_3_index, 1], alpha=.1, s=1, c='r')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i != 0:
            plt.axis("off")
    # plt.savefig(f"plots/diffusion_model.png", bbox_inches='tight',dpi=200)
    # plt.close()
    plt.show()


def plot_trail(n_datapoints,model):

    sample_evolution = model.denoise_sample(n_datapoints)
    sample_evolution = torch.stack(sample_evolution).detach().numpy()

    fig, ax = plt.subplots(1, 1)
    for i in range(n_datapoints):
        # Initialize a line for each particle
        plot_i = ax.plot(sample_evolution[:,i,0],  # first x coordinate
                         sample_evolution[:,i,1],  # first y coordinate
                         alpha=0.8, c="k")

    ax.scatter(sample_evolution[0,:,0], sample_evolution[0,:,1],alpha=0.8,color="g")
    ax.scatter(sample_evolution[-1, :, 0], sample_evolution[-1, :, 1], alpha=0.8, color="r")
    # Plot circle
    circle_1 = plt.Circle((1, 1),0.3,fill=False,linestyle="--")
    circle_2 = plt.Circle((-0.7, -0.3), 0.3, fill=False,linestyle="--")
    circle_3 = plt.Circle((-0.35, -1.5), 0.3, fill=False,linestyle="--")
    ax.add_artist(circle_1)
    ax.add_artist(circle_2)
    ax.add_artist(circle_3)


    plt.axis("equal")
    plt.show()



def animate_antidiffusion(n_datapoints,model,T):
    sample_evolution = model.denoise_sample(n_datapoints)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])

    particle_plot = plt.scatter(sample_evolution[0][:,0],sample_evolution[0][:,1],
                                alpha=0.2,marker=".")

    def update_plot(k):
        ax.set_title(f"k = {k}, t = {T - k}")
        particle_plot.set_offsets(sample_evolution[k].detach().numpy())
        return particle_plot

    anim = animation.FuncAnimation(fig,
                                   func=update_plot,
                                   # fargs=(list_of_trajectory_plots),
                                   frames=T,
                                   interval=50,
                                   blit=False)
    plt.show()
    return anim


def animate_antidiffusion_with_color(n_datapoints,model,T):
    sample_evolution = model.denoise_sample(n_datapoints)
    # Identify origins
    x_0 = sample_evolution[-1]
    point_1_index = (x_0[:, 0] > 0.2).nonzero().view(-1).detach().numpy()
    point_3_index = (x_0[:, 1] < -1).nonzero().view(-1).detach().numpy()
    point_2_index = np.setdiff1d(np.arange(n_datapoints), point_1_index)
    point_2_index = np.setdiff1d(point_2_index, point_3_index)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])

    particle_plot_1 = plt.scatter(sample_evolution[0][point_1_index,0],sample_evolution[0][point_1_index,1],alpha=0.2,marker=".",c="g")
    particle_plot_2 = plt.scatter(sample_evolution[0][point_2_index, 0], sample_evolution[0][point_2_index, 1], alpha=0.2, marker=".",c="b")
    particle_plot_3 = plt.scatter(sample_evolution[0][point_3_index, 0], sample_evolution[0][point_3_index, 1], alpha=0.2, marker=".",c="r")

    def update_plot(k):
        ax.set_title(f"k = {k}, t = {T - k}")
        particle_plot_1.set_offsets(sample_evolution[k][point_1_index,:].detach().numpy())
        particle_plot_2.set_offsets(sample_evolution[k][point_2_index, :].detach().numpy())
        particle_plot_3.set_offsets(sample_evolution[k][point_3_index, :].detach().numpy())
        return (particle_plot_1, particle_plot_2, particle_plot_3)

    anim = animation.FuncAnimation(fig,
                                   func=update_plot,
                                   # fargs=(list_of_trajectory_plots),
                                   frames=T,
                                   interval=150,
                                   blit=False)
    plt.show()
    return anim


def animate_antidiffusion_with_trail(n_datapoints,model,T):

    # Denoise random sample
    sample_evolution = model.denoise_sample(n_datapoints)
    sample_evolution = torch.stack(sample_evolution).detach().numpy()

    # Set up plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-3,3])
    ax.set_ylim([-3,3])

    list_of_trajectory_plots = []
    for i in range(n_datapoints):
        # Initialize a line for each particle
        plot_i = ax.plot(sample_evolution[0,i,0],  # first x coordinate
                         sample_evolution[0,i,1],  # first y coordinate
                         alpha=0.05, c="k")
        # Save the line plot
        list_of_trajectory_plots.append(plot_i[0])
    # Scatter plot of current position
    particle_plot = plt.scatter(sample_evolution[0,:,0],sample_evolution[0,:,1],
                                alpha=0.5,marker=".")

    TRAIL_HISTORY = 7
    def update_plot(k):
        ax.set_title(f"k = {k}, t = {T - k}")
        for i, plot_i in enumerate(list_of_trajectory_plots):
            plot_i.set_data(sample_evolution[max(0,k-TRAIL_HISTORY):k,i,0],sample_evolution[max(0,k-TRAIL_HISTORY):k,i,1])
        particle_plot.set_offsets(sample_evolution[k])
        return list_of_trajectory_plots + [particle_plot],

    anim = animation.FuncAnimation(fig,
                                   func=update_plot,
                                   frames=T,
                                   interval=75,
                                   blit=False)
    plt.show()
    return anim


def plot_beta_comparison(n_datapoints,model,T):
    k_frames = 10
    n_steps = T
    beta_configs = [
        0.005 * torch.ones(n_steps),
        0.05 * torch.ones(n_steps),
        torch.linspace(0.00001, 0.1, n_steps),
        torch.sigmoid(
            torch.linspace(-8, 5, n_steps)
        ) * (0.1 - 0.00001) + 0.00001,
    ]

    # Plot
    fig, axs = plt.subplots(nrows=len(beta_configs), ncols=1+k_frames+1)


    x0 = sample_batch(n_datapoints)

    for i, beta_config in enumerate(beta_configs):
        alpha = 1.0 - beta_config
        alpha_bar = torch.cumprod(alpha, dim=0)
        # Update model with new beta
        model.beta = beta_config
        model.alpha = alpha
        model.alpha_bar = alpha_bar
        # Diffuse the n timesteps
        data = [x0]
        for k in range(1,k_frames+1):
            _, _, x_k = model.diffuse_forward(x0=x0, t=int((n_steps / k_frames) * k))
            data.append(x_k)

        # Set up first column
        axs[i][0].plot(beta_config,color="k")
        axs[i][0].set_xticks([])
        axs[i][0].set_title(r"$\beta_{}$(t)".format(i))
        # Set up the rest of the columns
        for j, data_i in enumerate(data):
            axs[i][j+1].scatter(data_i[:, 0], data_i[:, 1], alpha=.01, s=1)
            axs[i][j+1].axis("equal")
            axs[i][j+1].axis("off")
            axs[i][j+1].set_xlim(-2,2)
            axs[i][j+1].set_ylim(-2, 2)

            if i == 0: # write time step
                axs[i][j+1].set_title(f'$t={int((n_steps / k_frames) * j)}$')
            if j == 10:
                mu_1, mu_2 = torch.mean(data_i,dim=0)
                sigma_1, sigma_2 = torch.std(data_i,dim=0)
                stats = (f"$\mu_x$, $\mu_y$ = ({round(mu_1.item(),2)},{round(mu_2.item(),2)})\n $\sigma_x$, $\sigma_y$ = ({round(sigma_1.item(),2)},{round(sigma_2.item(),2)})")
                bbox = dict(boxstyle='round', fc='white', ec='black', alpha=0.5)
                axs[i][j].text(2.5, 0, stats, fontsize=9, bbox=bbox,
                        transform=axs[i][j].transAxes, horizontalalignment='right')

    # plt.savefig(f"plots/diffusion_model.png", bbox_inches='tight')
    # plt.close()
    plt.show()
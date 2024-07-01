import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from vector_field import VectorField, right_up_flow
from particle_set import ParticleSet


class Simulation():
    def __init__(self, vector_field, particle_set):
        self.vector_field = vector_field
        self.particle_set = particle_set
        # self.kernel = kernel

    def run(self, dt, timesteps):
        for timestep_i in range(timesteps):
            self.particle_set.step(self.vector_field, dt)
            if (np.diag(self.particle_set.spatial_variance) < 0).any():
                break

    def plot(self):
        self.particle_set.plot_trajectory()
        self.vector_field.plot(-5, 15,-2.5, 2.5)

    def animate(self):

        #### Init trajectories ####
        n_timesteps = self.particle_set.timestep_counter
        trajectory_dt = self.particle_set.time_history[1] - self.particle_set.time_history[0]

        list_of_x_trajectories = self.particle_set.coordinates[0:n_timesteps + 1,:,0]
        list_of_y_trajectories = self.particle_set.coordinates[0:n_timesteps + 1,:,1]
        list_of_trajectory_plots = []

        # Plot
        fig, ax = plt.subplots(1, 1)
        x_min, x_max = np.min(list_of_x_trajectories), np.max(list_of_x_trajectories)
        y_min, y_max = np.min(list_of_y_trajectories), np.max(list_of_y_trajectories)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        for i in range(self.particle_set.n_particles):
            # Initialize all lines
            plot_i = ax.plot(self.particle_set.coordinates[0,i,0],
                             self.particle_set.coordinates[0,i,1],
                             alpha=0.2, c="k")
            # Save the line plot
            list_of_trajectory_plots.append(plot_i[0])

        #### Init quiver ####
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
        field_magnitude = self.vector_field.evaluate((X, Y, 0))
        U = field_magnitude[:, 0, :]
        V = field_magnitude[:, 1, :]
        quiver_plot = ax.quiver(X, Y, U, V)

        def update_plot(timestep_i):
            t_i = (timestep_i - 1) * trajectory_dt
            # timestep_i-1 because the effects of the vector field at t_i are not visible until t_i+1
            # Update quiver
            field_magnitude = self.vector_field.evaluate((X, Y, t_i))
            U = field_magnitude[:, 0, :]
            V = field_magnitude[:, 1, :]
            quiver_plot.set_UVC(U, V)
            # Update lines
            for i, plot_i in enumerate(list_of_trajectory_plots):
                plot_i.set_data(list_of_x_trajectories[0:timestep_i, i], list_of_y_trajectories[0:timestep_i, i])

            # ax.set_title(f"t = {t_i}")
            return quiver_plot, list_of_trajectory_plots,

        anim = animation.FuncAnimation(fig,
                                       func=update_plot,
                                       frames=np.arange(n_timesteps),
                                       interval=40,
                                       blit=False)
        fig.tight_layout()
        from PIL import Image
        anim.save('simulation.gif', writer='pillow')
        # plt.axis("equal")
        return anim

if __name__ == "__main__":

    # Vector field
    onades = VectorField(right_up_flow)
    # onades = VectorField(right_flow)
    # onades = VectorField(field_function_3)
    # Particle set
    pSet = ParticleSet()
    pSet.init_from_cloud(x=0, y=0, sigma=0.5, n_particles=70)
    # Simulate
    sim = Simulation(vector_field=onades, particle_set=pSet)
    sim.run(dt=0.1, timesteps=100)
    sim.plot()
    plt.show()

    sim.animate()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

MAX_PARTICLE_TIMESTEPS = 10000


class ParticleSet:
    def __init__(self, ):
        self.n_particles = None
        # Position
        self.x = None
        self.coordinates = None # x history
        # Time
        self.t = 0
        self.time_history = None
        # Mass
        self.mass = None
        self.com = None
        # Timestep counter
        self.timestep_counter = 0
        # Diffusion
        self.Kh = np.diag([0.01, 0.01])
        # self.Kh = np.diag([0.0, 0.0])
        self.spatial_variance = None

    def __getitem__(self, i):
        return self.coordinates[self.timestep_counter,i,:]

    def __len__(self):
        return self.n_particles

    def center_of_mass(self):
        """
        Uncomment the code below and change the parameter `weights`
        if the mass of the particles is different between them
        :return: numpy array
            returns the center of mass of the particle set at the most recent timestep
        """

        # mass_values = np.zeros(self.n_particles)
        # for i, particle_i in enumerate(self.particles):
        #     mass_values[i] = particle_i.mass

        return np.average(self.coordinates[self.timestep_counter,:,:],axis=0, weights=self.mass)

    def total_mass(self):
        """
        Computes the total mass of the particle set
        :return: float
        """
        return np.sum(self.mass)

    def init_from_cloud(self, x, variance, n_particles, t=0):
        """
        :param x: list, tuple, numpy array
            x and y coordinates of the center of the cloud
        :param variance: float
            variance for both x and y coordinates
        :param n_particles: int
            total particles to initialize
        """

        self.n_particles = n_particles

        # Create the matrix representation
        self.x = np.zeros((self.n_particles,2))
        self.coordinates = np.zeros((MAX_PARTICLE_TIMESTEPS, self.n_particles, 2))
        self.t = t
        self.time_history = np.zeros(MAX_PARTICLE_TIMESTEPS)
        self.mass = np.ones(self.n_particles) # all particles have the same mass

        # Create the particles
        for i in range(n_particles):
            coordinates_i = np.random.multivariate_normal(x, np.diag(variance * np.ones(2)))
            self.x[i,:] = coordinates_i
            self.coordinates[0,i,:] = coordinates_i

        self.com = np.average(self.coordinates[0, :, :], axis=0, weights=self.mass)
        self.spatial_variance = np.zeros((2,2))
        for i in range(n_particles):
            self.spatial_variance += (self.mass[i] / np.sum(self.mass)) * \
                                     (self.x[i] - self.com).reshape(2, 1) * (self.x[i] - self.com).reshape(1, 2)

    def init_from_existing_cloud(self, cloud, t=0):
        """
        :param cloud: numpy array
            (n_particles, 2) array of the coordinates of the particles
        :param t: float
            initial time
        """

        self.n_particles, _ = cloud.shape

        # Create the matrix representation
        self.x = np.zeros((self.n_particles,2))
        self.coordinates = np.zeros((MAX_PARTICLE_TIMESTEPS, self.n_particles, 2))
        self.t = t
        self.time_history = np.zeros(MAX_PARTICLE_TIMESTEPS)
        self.mass = np.ones(self.n_particles) # all particles have the same mass

        # Create the particles
        for i in range(self.n_particles):
            coordinates_i = cloud[i,:]
            self.x[i,:] = coordinates_i
            self.coordinates[0,i,:] = coordinates_i

        self.com = np.average(self.coordinates[0, :, :], axis=0, weights=self.mass)
        self.spatial_variance = np.zeros((2,2))
        for i in range(self.n_particles):
            self.spatial_variance += (self.mass[i] / np.sum(self.mass)) * \
                                     (self.x[i] - self.com).reshape(2, 1) * (self.x[i] - self.com).reshape(1, 2)

    def reset(self):
        # Position
        self.x = self.coordinates[0,:,:]
        self.coordinates = np.zeros((MAX_PARTICLE_TIMESTEPS, self.n_particles, 2))
        self.coordinates[0,:,:] = self.x
        # Time
        self.t = 0
        self.time_history = np.zeros(MAX_PARTICLE_TIMESTEPS)
        # Timestep counter
        self.timestep_counter = 0

    def step(self,vector_field, dt):
        if dt < 0:
            self.step_backwards(vector_field, dt)
        else:
            self.step_forward(vector_field, dt)

    def step_backwards(self, vector_field, dt):
        """
        Reversible-Time Particle Tracking Method
        :param vector_field: VectorField object
        :param dt: float
        """

        com_x, com_y = self.com
        com_x, com_y = com_x.reshape(1,1), com_y.reshape(1,1)
        u = vector_field.evaluate((com_x,
                                   com_y,
                                   self.t)
                                  ).reshape(2,)
        # Advection
        self.com += u * dt
        self.t += dt
        # Diffusion
        self.spatial_variance += 2 * dt * self.Kh
        if (np.diag(self.spatial_variance) < 0).any():
            print(self.timestep_counter, "spatial_variance=",self.spatial_variance)
        F = np.sqrt(np.diag(self.spatial_variance))
        R = np.random.normal(0,1,size=(self.n_particles,2))
        S = (np.random.randint(low=0,high=2,size=(self.n_particles,2)) * 2) - 1
        Q = R * S
        self.x = self.com + F * Q

        # Record history
        self.timestep_counter += 1
        self.coordinates[self.timestep_counter, :, :] = self.x
        self.time_history[self.timestep_counter] = self.t

    def step_forward(self, vector_field, dt):
        """
        Advection-diffusion timestep operations
        :param vector_field: VectorField object
        :param dt: float
        """
        u = vector_field.evaluate((self.coordinates[self.timestep_counter,:,0],
                                   self.coordinates[self.timestep_counter,:,1],
                                   self.t))

        # Advection
        self.x += u * dt
        self.t += dt

        # Diffusion
        dW = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=np.diag(dt * np.ones(2)),
            size=self.n_particles
        )
        self.x += np.sqrt(2 * np.diag(self.Kh)) * dW

        # Record history
        self.timestep_counter += 1
        self.coordinates[self.timestep_counter,:,:] = self.x
        self.time_history[self.timestep_counter] = self.t

    def plot(self):
        """
        Plots the particles at the most recent timestep
        """
        x_coordinates = self.coordinates[self.timestep_counter, :, 0]
        y_coordinates = self.coordinates[self.timestep_counter, :, 1]
        plt.scatter(x_coordinates, y_coordinates, alpha=0.2, c="b")

    def plot_trajectory(self):
        """
        Plots the trajectories of all the particles of the set
        """
        # Trajectory
        plt.plot(self.coordinates[0:self.timestep_counter+1, :, 0], self.coordinates[0:self.timestep_counter+1, :, 1],
                 c="k", alpha=0.2)
        # First and last coordinates
        plt.scatter(self.coordinates[0, :, 0], self.coordinates[0, :, 1], c="g")
        plt.scatter(self.coordinates[self.timestep_counter, :, 0], self.coordinates[self.timestep_counter, :, 1], c="r")

    def animate(self):

        # Coordinates of each particle on each timestep
        list_of_x_trajectories = self.coordinates[0:self.timestep_counter + 1,:,0]
        list_of_y_trajectories = self.coordinates[0:self.timestep_counter + 1,:,1]
        list_of_trajectory_plots = []

        fig, ax = plt.subplots(1, 1)
        ax.set_xlim([np.min(list_of_x_trajectories), np.max(list_of_x_trajectories)])
        ax.set_ylim([np.min(list_of_y_trajectories), np.max(list_of_y_trajectories)])

        for i in range(self.n_particles):
            # Initialize a plot for each particle
            plot_i = ax.plot(self.coordinates[0,i,0], # first x coordinate
                             self.coordinates[0,i,1], # first y coordinate
                             alpha=0.2, c="k")
            # Save the line plot
            list_of_trajectory_plots.append(plot_i[0])

        def update_plot(t):
            for i, plot_i in enumerate(list_of_trajectory_plots):
                plot_i.set_data(list_of_x_trajectories[0:t, i], list_of_y_trajectories[0:t, i])
            return list_of_trajectory_plots,

        anim = animation.FuncAnimation(fig,
                                       func=update_plot,
                                       # fargs=(list_of_trajectory_plots),
                                       frames=np.arange(self.timestep_counter),
                                       interval=50,
                                       blit=False)
        fig.tight_layout()
        # plt.show()
        return anim


if __name__ == "__main__":
    pSet = ParticleSet()
    pSet.init_from_cloud(x=(1,1),variance=0.1,n_particles=100)
    pSet.plot()
    plt.show()
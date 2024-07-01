import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def field_function_1(input_values):
    x, y, _ = input_values
    x_component = x / np.sqrt(np.power(x, 2) + np.power(y, 2))
    y_component = y / np.sqrt(np.power(x, 2) + np.power(y, 2))
    return np.stack((x_component, y_component), axis=1)

def field_function_2(input_values):
    x, y, t = input_values
    x_component = np.sin(t) * x / np.sqrt(np.power(x, 2) + np.power(y, 2))
    y_component = np.cos(t) * y / np.sqrt(np.power(x, 2) + np.power(y, 2))
    return np.stack((x_component, y_component), axis=1)


def field_function_3(input_values):
    x, y, _ = input_values
    x_component = np.sin(2) * x / np.sqrt(np.power(x, 2) + np.power(y, 2))
    y_component = np.cos(2) * y / np.sqrt(np.power(x, 2) + np.power(y, 2))
    return np.stack((x_component, y_component), axis=1)


def right_flow(input_values):
    x, y, _ = input_values
    x_component = x / x # = 1
    y_component = y * 0 # = 0
    return 1 * np.stack((x_component, y_component), axis=1)


def right_up_flow(input_values):
    x, y, t = input_values
    x_component = np.cos(t / 10) * x / x
    y_component = np.sin(t / 10) * y / y
    return np.stack((x_component, y_component), axis=1)


class VectorField:
    def __init__(self, function):
        self.field = function

    def evaluate(self, input_values):
        """
        Input should be a tuple: (x_coords, y_coords) or (x_coords, y_coords, time)
        """
        return self.field(input_values)

    def plot(self, x_min=-5, x_max=5, y_min=-5, y_max=5, t=0,a=1):
        # Generate meshgrid and compute the vector field on each coordinate
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 15), np.linspace(y_min, y_max, 15))
        field_magnitude = self.evaluate((X, Y, t))
        U = field_magnitude[:, 0, :]
        V = field_magnitude[:,1,:]
        # Plot
        # plt.figure()
        plt.quiver(X, Y, U, V,alpha=a)
        # plt.show()

    def animate(self, x_min=-5, x_max=5, y_min=-5, y_max=5, t_min=0, t_max=3):
        # Generate meshgrid and compute the vector field on each coordinate
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
        field_magnitude = self.evaluate((X, Y, t_min))
        U = field_magnitude[:, 0, :]
        V = field_magnitude[:, 1, :]
        # Plot
        fig, ax = plt.subplots(1, 1)
        quiver_plot = ax.quiver(X, Y, U, V)

        def update_quiver(t, quiver_plot, X, Y):
            field_magnitude = self.evaluate((X, Y, t))
            U = field_magnitude[:, 0, :]
            V = field_magnitude[:, 1, :]
            quiver_plot.set_UVC(U, V)
            return quiver_plot,

        anim = animation.FuncAnimation(fig,
                                       func=update_quiver,
                                       fargs=(quiver_plot, X, Y),
                                       frames=np.linspace(t_min, t_max, 80),
                                       interval=10,
                                       blit=False)
        fig.tight_layout()
        # plt.show()
        return anim


if __name__ == "__main__":
    a = VectorField(field_function_3)

    a.plot()
    plt.show()

    # a.animate()
    # plt.show()
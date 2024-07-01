# Matplotlib
from utils.locate_utils import read_csv_file
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import xarray

# Cartopy
# Cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature # pels features


def plot_traj_mitja(simulation_file_uri,  # or just the name
                    experimental_track_file):
    if experimental_track_file is not None:
        ################################
        #### Open experimental file ####
        ################################
        experimental_track_file = read_csv_file(experimental_track_file)

        # For subplot 1
        EXP_lonCoords, EXP_latCoords, EXP_timeCoords = experimental_track_file
        EXP_timeCoords = [np.datetime64(data) for data in EXP_timeCoords]

        # For subplot 2
        EXP_timeCoords = EXP_timeCoords - EXP_timeCoords[0]
        EXP_timeCoords = EXP_timeCoords.astype("float64")
        EXP_timeCoords = EXP_timeCoords * 1e-6 / (3600)  # estava en us
    else:
        EXP_lonCoords = []
        EXP_latCoords = []
        EXP_timeCoords = []

    #####################################
    #### Open simulation output file ####
    #####################################
    fitxer = xarray.open_dataset(simulation_file_uri)
    amount_particles = len(fitxer.traj)

    # Extract the average trajectory
    coordLonMitja = fitxer.lon.mean("traj")
    coordLatMitja = fitxer.lat.mean("traj")
    coordTime = fitxer.time[0].values
    print(coordTime.astype("float64") * 1e-9 / 3600)

    # Extract all the trajectories (one for each particle)
    ll_longituds = fitxer.lon.values
    ll_latituds = fitxer.lat.values
    ll_labels = [None for _ in range(amount_particles)]
    ll_labels[0] = "Simulated particles"  # for the legend

    # Set time as elapsed time
    coordTime = coordTime - coordTime[0]
    coordTime = coordTime.astype("float64")
    coordTime = coordTime * 1e-9 / (3600)  # hours

    # fisrt_n = 1000
    # plt.figure()
    # for nomLabel, longitud, latitud in zip(ll_labels, ll_longituds, ll_latituds):
    #     plt.scatter(longitud[:fisrt_n],
    #              latitud[:fisrt_n],
    #              c=coordTime[:fisrt_n],
    #              label=nomLabel,
    #              marker="o"
    #              )
    #     plt.plot(longitud[:fisrt_n],
    #              latitud[:fisrt_n],
    #              linestyle='--',
    #              alpha=0.5)
    # plt.colorbar()
    # plt.show()

    # Plotting bounding box
    margin = 1
    boundBox = (
        min(fitxer.lon.min("obs")) - margin,
        max(fitxer.lon.max("obs")) + margin,
        min(fitxer.lat.min("obs")) - margin,
        max(fitxer.lat.max("obs")) + margin,
    )

    # Create figure
    fig, ax_first_row = plt.subplots(1, 2,
                                     figsize=(13, 8),
                                     subplot_kw={'projection': ccrs.PlateCarree()})
    # fig, ax_first_row = plt.subplots(1, 2, figsize=(13, 8))

    # Configure cartopy on both subplots
    for ax_range in [0, 1]:
        ax_first_row[ax_range].set_extent([boundBox[0], boundBox[1], boundBox[2], boundBox[3]], ccrs.PlateCarree())
        ax_first_row[ax_range].add_feature(cfeature.RIVERS)
        ax_first_row[ax_range].add_feature(cfeature.LAND)
        ax_first_row[ax_range].add_feature(cfeature.OCEAN)
        ax_first_row[ax_range].add_feature(cfeature.COASTLINE)
        ax_first_row[ax_range].add_feature(cfeature.BORDERS, linestyle=":")
        map_grid = ax_first_row[ax_range].gridlines(draw_labels=True, zorder=-1)
        map_grid.top_labels = False
        map_grid.right_labels = False

        # with open("data/Barcelona_coastline.json", "r") as file:
        #     data = json.load(file)
        #
        # for _, val in data.items():
        #     ax_first_row[ax_range].plot(val["lon"], val["lat"], c="gray")
        #     ax_first_row[ax_range].axis("equal")

    ax_first_row[0].text((boundBox[0] + boundBox[1]) / 2, boundBox[2] * 1.001,
                         f"{amount_particles} particles simulated", horizontalalignment='center',
                         bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))

    # Subplot 1

    # Simulated trajectories
    for nomLabel, longitud, latitud in zip(ll_labels, ll_longituds, ll_latituds):
        ax_first_row[0].plot(longitud,
                             latitud,
                             linestyle='--',
                             c="r",
                             alpha=0.02,
                             label=nomLabel
                             )
        ax_first_row[0].scatter(longitud[:1],
                                latitud[:1],
                                c="g",
                                label=nomLabel,
                                marker="."
                                )
    # Average trajectory
    ax_first_row[0].plot(coordLonMitja,
                         coordLatMitja,
                         linestyle='-',
                         linewidth=1,
                         c="k",
                         alpha=1,
                         label="Average trajectory"
                         )

    # Trajectòria experimental
    if experimental_track_file is not None:
        ax_first_row[0].plot(EXP_lonCoords,
                             EXP_latCoords,
                             linestyle='-',
                             linewidth=1,
                             c="b",
                             alpha=1,
                             label="Experimental trajectory"
                             )
    ax_first_row[0].legend()

    # Subplot 2
    fused_lon = list(coordLonMitja.values) + list(EXP_lonCoords)
    fused_lat = list(coordLatMitja.values) + list(EXP_latCoords)
    fused_time = list(coordTime) + list(EXP_timeCoords)

    exp_size = 10
    sim_size = 100

    fused_size = [sim_size for _ in range(len(coordTime))] + [exp_size for _ in range(len(EXP_timeCoords))]

    # Experimental + simulated trajectory
    if experimental_track_file is not None:
        time_plot = ax_first_row[1].scatter(fused_lon,
                                            fused_lat,
                                            #                                             linestyle = '-',
                                            marker=".",
                                            #                                             linewidth = 3,
                                            #                                             c = "b",
                                            c=fused_time,
                                            s=fused_size,
                                            alpha=1,
                                            cmap="jet",
                                            )
        ax_first_row[1].scatter([], [], marker=".", c="b", alpha=1, s=sim_size, label="Simulated trajectory")
        ax_first_row[1].scatter([], [], marker=".", c="b", alpha=1, s=exp_size, label="Experimental trajectory")

    # Only simulated trajectory
    else:
        time_plot = ax_first_row[1].scatter(coordLonMitja,
                                            coordLatMitja,
                                            #                                         linestyle = '-',
                                            marker=".",
                                            s=sim_size,
                                            #                                         linewidth = 3,
                                            #                                         c = "k",
                                            c=coordTime,
                                            alpha=1,
                                            cmap="jet",
                                            label="Average trajectory"
                                            )
    ax_first_row[1].legend()

    # Colorbar
    cbar_ax = fig.add_axes([0.985,0.24, 0.026, 0.5])
    # new ax with dimensions of the colorbar

    # cbar = fig.colorbar(c, cax=cbar_ax)
    # barra = plt.colorbar(time_plot, ax=ax_first_row[1], location='right', shrink=0.8, pad=0.20)
    barra = plt.colorbar(time_plot, cax=cbar_ax)
    barra.ax.set_title('Elapsed hours')
    #     plt.savefig("holaaaaa.svg")
    # plt.tight_layout()

    return fig


def plot_all_trajectories(simulation_file_uri,starting_particle_distribution):

    #####################################
    #### Open simulation output file ####
    #####################################
    fitxer = xarray.open_dataset(simulation_file_uri)
    amount_particles = len(fitxer.traj)

    # Extract the average trajectory
    coordTime = fitxer.time[0].values

    # Extract all the trajectories (one for each particle)
    ll_longituds = fitxer.lon.values
    ll_latituds = fitxer.lat.values
    ll_labels = [None for _ in range(amount_particles)]
    ll_labels[0] = "Simulated particles"  # for the legend

    # Set time as elapsed time
    coordTime = coordTime - coordTime[0]
    coordTime = coordTime.astype("float64")
    coordTime = coordTime * 1e-9 / (3600)  # hours

    # Plotting bounding box
    margin = 0
    boundBox = (
        min(fitxer.lon.min("obs")) - margin,
        max(fitxer.lon.max("obs")) + margin,
        min(fitxer.lat.min("obs")) - margin,
        max(fitxer.lat.max("obs")) + margin,
    )

    # Create figure
    fig, ax_first_row = plt.subplots(1, 1,
                                     figsize=(7, 4),
                                     subplot_kw={'projection': ccrs.PlateCarree()})
    # Configure cartopy
    ax_first_row.set_extent([boundBox[0], boundBox[1], boundBox[2], boundBox[3]], ccrs.PlateCarree())
    ax_first_row.add_feature(cfeature.RIVERS)
    ax_first_row.add_feature(cfeature.LAND)
    ax_first_row.add_feature(cfeature.OCEAN)
    ax_first_row.add_feature(cfeature.COASTLINE)
    ax_first_row.add_feature(cfeature.BORDERS, linestyle=":")
    map_grid = ax_first_row.gridlines(draw_labels=True, zorder=-1)
    map_grid.top_labels = False
    map_grid.right_labels = False

    # with open("data/Barcelona_coastline.json", "r") as file:
    #     data = json.load(file)
    #
    # for _, val in data.items():
    #     ax_first_row[ax_range].plot(val["lon"], val["lat"], c="gray")
    #     ax_first_row[ax_range].axis("equal")

    # ax_first_row.text((boundBox[0] + boundBox[1]) / 2, boundBox[2] * 1.001,
    #                      f"{amount_particles} particles simulated", horizontalalignment='center',
    #                      bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))

    # Subplot 1
    # Plot initial particles
    ax_first_row.scatter(starting_particle_distribution[:,1],starting_particle_distribution[:,0],
                         c="k",label="starting",marker=".",alpha=0.2)
    # Simulated trajectories
    distinct_colors = generate_distinct_colors(len(ll_labels))

    # analisis_lon = []
    # analisis_lat = []
    # print(ll_longituds)
    # print(len(ll_longituds))
    # aa = input("jgrkf")
    for nomLabel, longitud, latitud, color_i in zip(ll_labels, ll_longituds, ll_latituds, distinct_colors):
        # Plot variance
        variance_evolution = get_variance_evolution(0.1/len(ll_labels),longitud)
        print("variance_evolution=",variance_evolution)
        R, G, B = color_i
        for i in range(0,len(longitud),20):
            drawing_colored_circle = plt.Circle((longitud[i], latitud[i]), variance_evolution[i],
                                                alpha=0.01,color=(R/255,G/255,B/255)) # alpha=0.005
            # axes.set_aspect(1)
            ax_first_row.add_artist(drawing_colored_circle)
        # Plot trajectory
        ax_first_row.plot(longitud,
                          latitud,
                          linestyle='--',
                          c="r",
                          alpha=0.2,
                          label=nomLabel
                          )
        # Plot starting points
        ax_first_row.scatter(longitud[0],latitud[0],c="g",label=nomLabel,marker="o")
        # Plot ending points
        ax_first_row.scatter(longitud[-1], latitud[-1], c="r", label=nomLabel, marker="o")
        # if longitud[-1] > 2:
        #     analisis_lon.append(longitud[-1])
        #     analisis_lat.append(latitud[-1])

    # print("mean")
    # print(sum(analisis_lon)/len(analisis_lon))
    # print(sum(analisis_lat) / len(analisis_lat))
    # print("var")
    # print(np.var(analisis_lon))
    # print(np.var(analisis_lat))
    # plt.figure()
    # plt.scatter(analisis_lon,analisis_lat,c="r",label="Simulated cloud",zorder=10,alpha=0.9)
    # X, Y = 40.6, 2.44
    # VARIANCE = 0.001
    # mat = np.random.multivariate_normal([X, Y], np.diag(VARIANCE * np.ones(2)), size=110)
    # plt.scatter(mat[:,1],mat[:,0],c="b",label="Original cloud",zorder=10,alpha=0.9)
    # plt.axis("equal")
    # plt.grid(zorder=-10)
    # plt.legend()
    # plt.show()
    # ax_first_row.legend()
    ax_first_row.axis("equal")
    # fig.tight_layout()
    return fig


def get_variance_evolution(initial_variance,longitud):
    variance_evolution = [initial_variance]
    D = 0.000001
    # D = 0.0000001
    D = 2.25e-11
    # print("\nNOU")
    for i in range(len(longitud)-1):
        # aa = input("fjkdl")
        last_variance = variance_evolution[-1]
        # print("last_variance=",last_variance)
        new_variance = last_variance - 2 * D * 5 * 60
        # print("new_variance=", new_variance)
        if new_variance < 0:
            # print("Ep! new_variance=", new_variance)
            variance_evolution.append(0)
        else:
            variance_evolution.append(new_variance)
        # variance_evolution.append(new_variance)

    # variance_evolution.reverse()
    # variance_evolution = [0.001 for i in range(len(longitud))]
    return np.array(variance_evolution)


def generate_distinct_colors(N):
    """
    Generate N most distinct colors in RGB.

    Parameters:
    N (int): Number of distinct colors to generate.

    Returns:
    list of tuples: List of RGB color tuples.
    """
    # Generate evenly spaced hues
    hues = np.linspace(0, 1, N, endpoint=False)

    # Set saturation and value to maximum
    saturation = 1
    value = 1

    # Convert HSV to RGB
    colors = [hsv_to_rgb((hue, saturation, value)) for hue in hues]

    # Convert colors from floats [0, 1] to integers [0, 255]
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    return colors_rgb


if __name__ == "__main__":
    AMOUNT_OF_PARTICLES = 500
    X, Y = 40.6, 2.44
    VARIANCE = 0.001
    mat = np.random.multivariate_normal([X, Y], np.diag(VARIANCE * np.ones(2)), size=AMOUNT_OF_PARTICLES)

    fig = plot_all_trajectories(simulation_file_uri="file_logging/simulation_output_file/2_clusters.nc",
                                starting_particle_distribution=mat)
    plt.show()
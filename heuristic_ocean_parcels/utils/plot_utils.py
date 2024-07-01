import logging

logger = logging.getLogger(__name__)

from parcels import FieldSet, NestedField
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from parcels.plotting import create_parcelsfig_axis, cartopy_colorbar, parsetimestr
import cartopy.crs as ccrs
from parcels.field import Field, VectorField
from parcels.grid import GridCode, CurvilinearGrid
from parcels.tools.statuscodes import TimeExtrapolationError
import cmocean
import copy
import config as cfg
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.animation import FuncAnimation, FFMpegWriter
from math import radians, cos, sin, asin, sqrt
from matplotlib import cm
from scipy.interpolate import interpn
from matplotlib.colors import Normalize
import xarray


def plot_distancia_viajada(datos_xr):
    """
    Mostra dues gràfiques de la distancia viatjada per les particules.
    La primera els eixos son diatancia(m) x num_observació i la segona es distancia x temps
    """
    x = datos_xr.lon.values
    y = datos_xr.lat.values
    distance = np.cumsum(np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y))), axis=1)
    real_time = datos_xr.time
    # Distancia x Observaciones
    time_since_release = (real_time.values.transpose() - real_time.values[:, 0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax1.set_ylabel('Distance travelled [m]')
    ax1.set_xlabel('observation', weight='bold')
    # Distancia x Tiempo
    d_plot = ax1.plot(distance.transpose())
    ax2.set_ylabel('Distance travelled [m]')
    ax2.set_xlabel('time', weight='bold')
    d_plot_t = ax2.plot(real_time.T[1:], distance.transpose())
    plt.show()


def plot_trajectory_basic(datos_xr):
    """
    Mostra exemples de diferents opcions de dibuixat de trajectories que es poden fer.
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    ###-Points-###
    ax1.set_title('Points')
    ax1.scatter(datos_xr['lon'].T, datos_xr['lat'].T)
    ###-Lines-###
    ax2.set_title('Lines')
    ax2.plot(datos_xr['lon'].T, datos_xr['lat'].T)
    ###-Points + Lines-###
    ax3.set_title('Points + Lines')
    ax3.plot(datos_xr['lon'].T, datos_xr['lat'].T, marker='o')
    ###-Not Transposed-###
    ax4.set_title('Not transposed')
    ax4.plot(datos_xr['lon'], datos_xr['lat'], marker='o')
    plt.show()


def plot_animate_basic(data, domain={'N': 42.0, 'S': 41, 'E': 3.1, 'W': 1.8}):
    """
    Plot d'animació basica (sense fieldset) del moviment de les particules
    """
    from matplotlib.animation import FuncAnimation
    from datetime import timedelta as delta
    outputdt = delta(hours=1)
    timerange = np.arange(np.nanmin(data['time'].values),
                          np.nanmax(data['time'].values) + np.timedelta64(outputdt),
                          outputdt)  # timerange in nanoseconds
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    ax = fig.add_subplot()

    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    ax.set_xlim(domain['W'], domain['E'])
    ax.set_ylim(domain['N'], domain['S'])
    # plt.xticks(rotation=45)
    time_id = np.where(data['time'] == timerange[0])  # Indices of the data where time = 0
    scatter = ax.scatter(data['lon'].values[time_id], data['lat'].values[time_id])
    t = np.datetime_as_string(timerange[0], unit='h')
    title = ax.set_title('Particles at t = ' + t)

    def animate(i):
        t = np.datetime_as_string(timerange[i], unit='h')
        title.set_text('Particles at t = ' + t)
        time_id = np.where(data['time'] == timerange[i])
        scatter.set_offsets(np.c_[data['lon'].values[time_id], data['lat'].values[time_id]])

    anim = FuncAnimation(fig, animate, frames=len(timerange), interval=500)
    plt.show()


def plot_cartopy_first_frame(dataset, domain):
    # projection = ccrs.EuroPP()  # no funcionan algunas cosas del plotfield (como las etiquetas de ejes)
    projection = ccrs.PlateCarree()
    field = dataset.fieldset.UV
    if type(field) is VectorField:
        spherical = True if field.U.grid.mesh == 'spherical' else False
        field = [field.U, field.V]
        plottype = 'vector'
    elif type(field) is Field:
        spherical = True if field.grid.mesh == 'spherical' else False
        field = [field]
        plottype = 'scalar'
    else:
        raise RuntimeError('field needs to be a Field or VectorField object')
    show_land = True
    plt, fig, ax, cartopy = create_parcelsfig_axis(spherical, show_land, projection=projection)
    ax.coastlines()
    plt.text(2.1513, 41.3358, 'Barcelona Harbour', horizontalalignment='right', transform=ccrs.Geodetic(),
             bbox=dict(boxstyle='round', ec=(0.5, 0.5, 0.8), fc=(0.5, 0.5, 1.0)))
    plt.show()


def parsedomain(domain, field):
    field.grid.check_zonal_periodic()
    dominio_desplazado = False
    if domain is not None:
        if not isinstance(domain, dict) and len(domain) == 4:  # for backward compatibility with <v2.0.0
            new_domain = {'N': domain[0], 'S': domain[1], 'E': domain[2], 'W': domain[3]}
        else:
            new_domain = domain.copy()
        min_lon, max_lon = field.grid.lon[0], field.grid.lon[-1]
        min_lat, max_lat = field.grid.lat[0], field.grid.lat[-1]
        if new_domain['W'] < min_lon:
            dominio_desplazado = True
            new_domain['W'] = min_lon
        if new_domain['E'] > max_lon:
            dominio_desplazado = True
            new_domain['E'] = max_lon
        if new_domain['S'] < min_lat:
            dominio_desplazado = True
            new_domain['S'] = min_lat
        if new_domain['N'] > max_lat:
            dominio_desplazado = True
            new_domain['N'] = max_lat
        _, _, _, lonW, latS, _ = field.search_indices(new_domain['W'], new_domain['S'], 0, 0, 0, search2D=True)
        _, _, _, lonE, latN, _ = field.search_indices(new_domain['E'], new_domain['N'], 0, 0, 0, search2D=True)
        return latN + 1, latS, lonE + 1, lonW, dominio_desplazado
    else:
        if field.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            return field.grid.lon.shape[0], 0, field.grid.lon.shape[1], 0, dominio_desplazado
        else:
            return len(field.grid.lat), 0, len(field.grid.lon), 0, dominio_desplazado


def plotfield(field, show_time=None, domain=None, depth_level=0, projection=None, land=True,
              vmin=None, vmax=None, savefile=None, densidad_flechas=1.0, **kwargs):
    """Function to plot a Parcels Field

    :param show_time: Time at which to show the Field
    :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
    :param depth_level: depth level to be plotted (default 0)
    :param projection: type of cartopy projection to use (default PlateCarree)
    :param land: Boolean whether to show land. This is ignored for flat meshes
    :param vmin: minimum colour scale (only in single-plot mode)
    :param vmax: maximum colour scale (only in single-plot mode)
    :param savefile: Name of a file to save the plot to
    :param animation: Boolean whether result is a single plot, or an animation
    """

    if type(field) is VectorField:
        spherical = True if field.U.grid.mesh == 'spherical' else False
        field = [field.U, field.V]
        plottype = 'vector'
    elif type(field) is Field:
        spherical = True if field.grid.mesh == 'spherical' else False
        field = [field]
        plottype = 'scalar'
    else:
        raise RuntimeError('field needs to be a Field or VectorField object')

    if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.CurvilinearSGrid]:
        logger.warning('Field.show() does not always correctly determine the domain for curvilinear grids. '
                       'Use plotting with caution and perhaps use domain argument as in the NEMO 3D tutorial')

    plt, fig, ax, cartopy = create_parcelsfig_axis(spherical, land, projection=projection,
                                                   cartopy_features=kwargs.pop('cartopy_features', []))
    if plt is None:
        return None, None, None, None  # creating axes was not possible

    data = {}
    plotlon = {}
    plotlat = {}
    for i, fld in enumerate(field):
        show_time = fld.grid.time[0] if show_time is None else show_time
        if fld.grid.defer_load:
            fld.fieldset.computeTimeChunk(show_time, 1)
        (idx, periods) = fld.time_index(show_time)
        show_time -= periods * (fld.grid.time_full[-1] - fld.grid.time_full[0])
        if show_time > fld.grid.time[-1] or show_time < fld.grid.time[0]:
            raise TimeExtrapolationError(show_time, field=fld, msg='show_time')

        latN, latS, lonE, lonW, dominio_desplazado = parsedomain(domain, fld)
        if isinstance(fld.grid, CurvilinearGrid):
            plotlon[i] = fld.grid.lon[latS:latN, lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN, lonW:lonE]
        else:
            plotlon[i] = fld.grid.lon[lonW:lonE]
            plotlat[i] = fld.grid.lat[latS:latN]
        if i > 0 and not np.allclose(plotlon[i], plotlon[0]):
            raise RuntimeError('VectorField needs to be on an A-grid for plotting')
        if fld.grid.time.size > 1:
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[depth_level, latS:latN,
                          lonW:lonE]
            else:
                data[i] = np.squeeze(fld.temporal_interpolate_fullfield(idx, show_time))[latS:latN, lonW:lonE]
        else:
            if fld.grid.zdim > 1:
                data[i] = np.squeeze(fld.sim_data)[depth_level, latS:latN, lonW:lonE]
            else:
                data[i] = np.squeeze(fld.sim_data)[latS:latN, lonW:lonE]

    if plottype == 'vector':
        if field[0].interp_method == 'cgrid_velocity':
            logger.warning_once(
                'Plotting a C-grid velocity field is achieved via an A-grid projection, reducing the plot accuracy')
            d = np.empty_like(data[0])
            d[:-1, :] = (data[0][:-1, :] + data[0][1:, :]) / 2.
            d[-1, :] = data[0][-1, :]
            data[0] = d
            d = np.empty_like(data[0])
            d[:, :-1] = (data[0][:, :-1] + data[0][:, 1:]) / 2.
            d[:, -1] = data[0][:, -1]
            data[1] = d

        spd = data[0] ** 2 + data[1] ** 2
        speed = np.where(spd > 0, np.sqrt(spd), 0)
        vmin = speed.min() if vmin is None else vmin
        vmax = speed.max() if vmax is None else vmax
        # ncar_cmap = copy.copy(cmocean.cm.speed)  # gradiente verde
        ncar_cmap = copy.copy(cmocean.cm.matter)  # gradiente granate
        # ncar_cmap = copy.copy(plt.cm.gist_ncar)  # gradiente multicolor
        ncar_cmap.set_over('k')
        ncar_cmap.set_under('w')
        if isinstance(field[0].grid, CurvilinearGrid):
            x, y = plotlon[0], plotlat[0]
        else:
            x, y = np.meshgrid(plotlon[0], plotlat[0])
        u = np.where(speed > 0., data[0] / speed, np.nan)
        v = np.where(speed > 0., data[1] / speed, np.nan)
        skip = (slice(None, None, int(1 / densidad_flechas)), slice(None, None, int(1 / densidad_flechas)))
        if cartopy:
            cs = ax.quiver(np.asarray(x)[skip], np.asarray(y)[skip], np.asarray(u)[skip], np.asarray(v)[skip],
                           speed[skip],
                           cmap=ncar_cmap, clim=[vmin, vmax], scale=100, transform=cartopy.crs.PlateCarree(),
                           minshaft=3, pivot='mid')
        else:
            cs = ax.quiver(x[skip], y[skip], u[skip], v[skip], speed[skip], cmap=ncar_cmap, clim=[vmin, vmax], scale=50)
    else:
        vmin = data[0].min() if vmin is None else vmin
        vmax = data[0].max() if vmax is None else vmax
        # pc_cmap = copy.copy(plt.cm.get_cmap('viridis'))
        pc_cmap = copy.copy(cmocean.cm.speed)
        pc_cmap.set_over('k')
        pc_cmap.set_under('w')
        assert len(data[0].shape) == 2
        if field[0].interp_method == 'cgrid_tracer':
            d = data[0][1:, 1:]
        elif field[0].interp_method == 'cgrid_velocity':
            if field[0].fieldtype == 'U':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][1:, :-1] + data[0][1:, 1:]) / 2.
            elif field[0].fieldtype == 'V':
                d = np.empty_like(data[0])
                d[:-1, :-1] = (data[0][:-1, 1:] + data[0][1:, 1:]) / 2.
            else:  # W
                d = data[0][1:, 1:]
        else:  # if A-grid
            d = (data[0][:-1, :-1] + data[0][1:, :-1] + data[0][:-1, 1:] + data[0][1:, 1:]) / 4.
            d = np.where(data[0][:-1, :-1] == 0, np.nan, d)
            d = np.where(data[0][1:, :-1] == 0, np.nan, d)
            d = np.where(data[0][1:, 1:] == 0, np.nan, d)
            d = np.where(data[0][:-1, 1:] == 0, np.nan, d)
        if cartopy:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], d, cmap=pc_cmap, transform=cartopy.crs.PlateCarree())
        else:
            cs = ax.pcolormesh(plotlon[0], plotlat[0], d, cmap=pc_cmap)

    if cartopy is None:
        ax.set_xlim(np.nanmin(plotlon[0]), np.nanmax(plotlon[0]))
        ax.set_ylim(np.nanmin(plotlat[0]), np.nanmax(plotlat[0]))
    elif domain is not None:
        ax.set_extent([np.nanmin(plotlon[0]), np.nanmax(plotlon[0]), np.nanmin(plotlat[0]), np.nanmax(plotlat[0])],
                      crs=cartopy.crs.PlateCarree())
    if dominio_desplazado:
        # aqui se puede cambiar el dominio mostrado (mas allá de donde haya datos)
        new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
        ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
        # ax.set_extent([np.nanmin(plotlon[0]-2), np.nanmax(plotlon[0]+2), np.nanmin(plotlat[0]-2), np.nanmax(plotlat[0])+2], crs=cartopy.crs.PlateCarree())
    cs.set_clim(vmin, vmax)

    cartopy_colorbar(cs, plt, fig, ax)

    # Labels, etc
    timestr = parsetimestr(field[0].grid.time_origin, show_time)
    titlestr = kwargs.pop('titlestr', '')
    if field[0].grid.zdim > 1:
        if field[0].grid.gtype in [GridCode.CurvilinearZGrid, GridCode.RectilinearZGrid]:
            gphrase = 'depth'
            depth_or_level = field[0].grid.depth[depth_level]
        else:
            gphrase = 'level'
            depth_or_level = depth_level
        depthstr = ' at %s %g ' % (gphrase, depth_or_level)
    else:
        depthstr = ''
    if plottype == 'vector':
        ax.set_title(titlestr + 'Velocity field' + depthstr + timestr)
    else:
        ax.set_title(titlestr + field[0].name + depthstr + timestr)

    if not spherical:
        ax.set_xlabel('Zonal distance [m]')
        ax.set_ylabel('Meridional distance [m]')

    plt.draw()

    if savefile:
        plt.savefig(savefile)
        logger.info('Plot saved to ' + savefile + '.png')
        plt.close()

    return plt, fig, ax, cartopy



def plot_trajectories(sim_data, field=None, domain=None, terrain=None, terrain_zoom=10, densidad_flechas=1 / 16,
                      flechas_en_paths=True, graph_type='lines'):
    land = True if terrain is None else False
    if field is not None:
        # Con el field por debajo de las particulas
        plt, fig, ax, cartopy = plotfield(field, domain=domain, titlestr="Trayectorias ", land=land,
                                          densidad_flechas=densidad_flechas)  # vectores
    else:
        # Con fondo maritimo homogeneo
        plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
        if domain is not None:
            new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
            ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
    # set figure size
    fig.set_size_inches(13, 6)

    lon = np.ma.filled(sim_data.variables['lon'], np.nan)
    lat = np.ma.filled(sim_data.variables['lat'], np.nan)
    t = np.ma.filled(sim_data.variables['time'], np.nan)
    mesh = sim_data.attrs['parcels_mesh'] if 'parcels_mesh' in sim_data.attrs else 'spherical'
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Para usar el mismo mapa de colores en plot que en scatter tengo que convertir el mapa a rgb
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    # if graph_type=='points':
    #     ######
    #     outputdt = timedelta(hours=1)
    #     timerange = np.arange(np.nanmin(sim_data['time'].values),
    #                           np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
    #                           outputdt)
    #     for idt in range(len(timerange)):
    #         time_id = np.where(sim_data['time'] == timerange[idt])
    #         ax.scatter(sim_data['lon'].values[time_id], sim_data['lat'].values[time_id], s=5)
    #     ######
    # else:
    #     ### Trayectorias con flechas
    logger.info(f"Plotting trajectories using lines {'and arrows' if flechas_en_paths else ''}")
    for idx in color_index:
        colorVal = scalarMap.to_rgba(idx)
        ax.plot(lon[idx], lat[idx], '-', color=colorVal, lw=1)
        if flechas_en_paths:
            for i in range(len(lon[idx]) - 1):
                plt.arrow(lon[idx, i], lat[idx, i], (lon[idx, i + 1] - lon[idx, i]) / 2,
                          (lat[idx, i + 1] - lat[idx, i]) / 2, shape='full', length_includes_head=False, lw=0,
                          width=0.0005, color=colorVal)

    # Mostrar la grafica
    ax.text(-0.1, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    ax.text(0.5, -0.12, 'Longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    plt.show()
    plt.pause(1)


# fig.savefig('Images/DispersionSimulation.tiff')


def plot_animation(sim_data, field=None, domain=None, terrain=None, terrain_zoom=10, densidad_flechas=1 / 16,
                   filename=None, show_plot=False):
    land = True if terrain is None else False
    if field is not None:
        # Con el field por debajo de las particulas
        plt, fig, ax, cartopy = plotfield(field, domain=domain, titlestr="Trayectorias ", land=land,
                                          densidad_flechas=densidad_flechas)  # vectores
    else:
        # Con fondo maritimo homogeneo
        plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
        if domain is not None:
            new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
            ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
    # set figure size
    fig.set_size_inches(13, 6)

    lon = np.ma.filled(sim_data.variables['lon'], np.nan)
    lat = np.ma.filled(sim_data.variables['lat'], np.nan)
    t = np.ma.filled(sim_data.variables['time'], np.nan)
    mesh = sim_data.attrs['parcels_mesh'] if 'parcels_mesh' in sim_data.attrs else 'spherical'
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Para usar el mismo mapa de colores en plot que en scatter tengo que convertir el mapa a rgb
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    outputdt = timedelta(hours=1)
    timerange = np.arange(np.nanmin(sim_data['time'].values),
                          np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                          outputdt)
    time_id = np.where(sim_data['time'] == timerange[0])  # Indices of the data where time = 0
    scatter = ax.scatter(sim_data['lon'].values[time_id], sim_data['lat'].values[time_id], s=10)
    t = np.datetime_as_string(timerange[0], unit='h')
    title = ax.set_title('Particles at t = ' + t)

    def animate(i):
        t = np.datetime_as_string(timerange[i], unit='h')
        title.set_text('Particles at t = ' + t)
        time_id = np.where(sim_data['time'] == timerange[i])
        scatter.set_offsets(np.c_[sim_data['lon'].values[time_id], sim_data['lat'].values[time_id]])

    anim = FuncAnimation(fig, animate, frames=len(timerange), interval=500)
    if show_plot:
        plt.show()
    if filename is not None:
        logger.info(f"Saving animation video to {filename}")
        writervideo = FFMpegWriter(fps=30)
        anim.save(filename, writer=writervideo)
        logger.info("Video saved")


def plot_hist2d_video(sim_data, domain=None, terrain=None, terrain_zoom=10, densidad_flechas=1 / 16, filename=None,
                      show_plot=False):
    land = True if terrain is None else False
    # Con fondo maritimo homogeneo
    plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
    if domain is not None:
        new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
        ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
        # ax.coastlines()
        ax.stock_img()
    else:
        # land = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
        # land = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',facecolor=cartopy.feature.COLORS['land'])
        # ax.add_feature(cartopy.feature.LAND)
        pass

    # set figure size
    fig.set_size_inches(13, 6)

    lon = np.ma.filled(sim_data.variables['lon'], np.nan)
    lat = np.ma.filled(sim_data.variables['lat'], np.nan)
    t = np.ma.filled(sim_data.variables['time'], np.nan)
    mesh = sim_data.attrs['parcels_mesh'] if 'parcels_mesh' in sim_data.attrs else 'spherical'
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Para usar el mismo mapa de colores en plot que en scatter tengo que convertir el mapa a rgb
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    outputdt = timedelta(days=1)
    timerange = np.arange(np.nanmin(sim_data['time'].values),
                          np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                          outputdt)
    time_id = np.where(np.logical_and(sim_data['time'] >= timerange[0],
                                      sim_data['time'] <= timerange[1]))  # Indices of the data where time = 0
    lon = sim_data['lon'].values[time_id]
    lat = sim_data['lat'].values[time_id]
    lon = lon.flatten()
    lon = lon[np.logical_not(np.isnan(lon))]
    lat = lat.flatten()
    lat = lat[np.logical_not(np.isnan(lat))]
    # hist = ax.hist2d(lon, lat, bins=nbins)
    width_km = haversine(domain["W"], domain["N"], domain["E"], domain["N"])
    height_km = haversine(domain["W"], domain["N"], domain["W"], domain["S"])
    print(f'Map width={width_km:.3f} km')
    print(f'Map height={height_km:.3f} km')
    nbins = (round(width_km), round(height_km))
    # nbins=(round(abs(domain['W']-domain['E'])*factor_bins), round(abs(domain['S']-domain['N'])*factor_bins))
    print(f'Number of bins for 2d histogram: {nbins}')
    data, x, y = np.histogram2d(lon, lat, bins=nbins, range=[[domain['W'], domain['E']], [domain['S'], domain['N']]])
    # scatter = ax.scatter(x, y, c=data)
    alpha = (data != 0)
    alpha = alpha.astype(float)
    im = plt.imshow(data.T,
                    interpolation='gaussian',
                    origin='upper',
                    extent=(domain['W'], domain['E'], domain['N'], domain['S']),
                    alpha=alpha.T,
                    zorder=100,
                    cmap='jet')
    # plt.xticks(list(range(len(x))), x)
    # plt.yticks(list(range(len(y))), y)

    # scatter = ax.scatter(sim_data['lon'].values[time_id], sim_data['lat'].values[time_id], s=10)
    t = np.datetime_as_string(timerange[0], unit='h')
    title = ax.set_title('Particles density at t = ' + t)

    def animate(i):
        t = np.datetime_as_string(timerange[i], unit='h')
        title.set_text('Particles density at t = ' + t)
        time_id = np.where(np.logical_and(sim_data['time'] >= timerange[i],
                                          sim_data['time'] <= timerange[i + 1]))  # Indices of the data where time = 0
        lon = sim_data['lon'].values[time_id]
        lat = sim_data['lat'].values[time_id]
        lon = lon.flatten()
        lon = lon[np.logical_not(np.isnan(lon))]
        lat = lat.flatten()
        lat = lat[np.logical_not(np.isnan(lat))]
        data, x, y = np.histogram2d(lon, lat, bins=nbins,
                                    range=[[domain['W'], domain['E']], [domain['S'], domain['N']]])
        # range define the extension of the grid, data = nº of points per bin, we need the bin correspondance to km2
        alpha = (data != 0)
        alpha = alpha.astype(float)
        im.set_data(data.T)
        im.set_alpha(alpha.T)
        # im = plt.imshow(data, interpolation='gaussian', origin='upper',
        #                 extent=(domain['E'], domain['W'], domain['S'], domain['N']), alpha=alpha)
        # scatter.set_offsets(x, y, )
        # hist = ax.hist2d(lon, lat, bins=nbins)

    anim = FuncAnimation(fig, animate, frames=len(timerange) - 1, interval=500)
    if show_plot:
        plt.show()
    if filename is not None:
        logger.info(f"Saving animation video to {filename}")
        writervideo = FFMpegWriter(fps=30)
        anim.save(filename, writer=writervideo)
        logger.info("Video saved")


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def plot_hist2d(sim_data, domain=None, terrain=None, terrain_zoom=10, densidad_flechas=1 / 16):
    land = True if terrain is None else False
    # Con fondo maritimo homogeneo
    plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
    if domain is not None:
        new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
        ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
        # ax.coastlines()
        ax.stock_img()
    else:
        # land = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
        # land = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face',facecolor=cartopy.feature.COLORS['land'])
        # ax.add_feature(cartopy.feature.LAND)
        pass

    # set figure size
    fig.set_size_inches(13, 6)

    lon = np.ma.filled(sim_data.variables['lon'], np.nan)
    lat = np.ma.filled(sim_data.variables['lat'], np.nan)
    t = np.ma.filled(sim_data.variables['time'], np.nan)
    mesh = sim_data.attrs['parcels_mesh'] if 'parcels_mesh' in sim_data.attrs else 'spherical'
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Para usar el mismo mapa de colores en plot que en scatter tengo que convertir el mapa a rgb
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    outputdt = timedelta(days=1)
    timerange = np.arange(np.nanmin(sim_data['time'].values),
                          np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                          outputdt)
    # time_id = np.where(np.logical_and(sim_data['time'] >= timerange[0], sim_data['time'] <= timerange[-1]))  # Indices of the data where time = 0
    time_id = np.where(np.logical_and(sim_data['time'] >= timerange[-2],
                                      sim_data['time'] <= timerange[-1]))  # Indices of the data where time = 0
    lon = sim_data['lon'].values[time_id]
    lat = sim_data['lat'].values[time_id]
    lon = lon.flatten()
    lon = lon[np.logical_not(np.isnan(lon))]
    lat = lat.flatten()
    lat = lat[np.logical_not(np.isnan(lat))]
    # hist = ax.hist2d(lon, lat, bins=nbins)
    width_km = haversine(domain["W"], domain["N"], domain["E"], domain["N"])
    height_km = haversine(domain["W"], domain["N"], domain["W"], domain["S"])
    print(f'Map width={width_km:.3f} km')
    print(f'Map height={height_km:.3f} km')
    nbins = (round(width_km), round(height_km))
    # nbins=(round(abs(domain['W']-domain['E'])*factor_bins), round(abs(domain['S']-domain['N'])*factor_bins))
    print(f'Number of bins for 2d histogram: {nbins}')
    data, x, y = np.histogram2d(lon, lat, bins=nbins, range=[[domain['W'], domain['E']], [domain['S'], domain['N']]])
    # scatter = ax.scatter(x, y, c=data)
    alpha = (data != 0)
    alpha = alpha.astype(float)
    im = plt.imshow(data.T,
                    interpolation='gaussian',
                    origin='upper',
                    extent=(domain['W'], domain['E'], domain['N'], domain['S']),
                    alpha=alpha.T,
                    zorder=100,
                    cmap='jet')

    # t = np.datetime_as_string(timerange[0], unit='h')
    t = np.datetime_as_string(timerange[-1], unit='h')
    title = ax.set_title('Particles density at t = ' + t)
    plt.show()
    # fig.savefig('heatmap.pdf')


def plot_hist2d_v2(sim_data, domain=None, terrain=None, terrain_zoom=10, sim_steps=24):
    land = True if terrain is None else False
    # Con fondo maritimo homogeneo
    plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
    if domain is not None:
        new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
        ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
        # ax.coastlines()
        ax.stock_img()

    # set figure size
    fig.set_size_inches(13, 6)

    lon = np.ma.filled(sim_data.variables['lon'], np.nan)
    lat = np.ma.filled(sim_data.variables['lat'], np.nan)
    t = np.ma.filled(sim_data.variables['time'], np.nan)
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Para usar el mismo mapa de colores en plot que en scatter tengo que convertir el mapa a rgb
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    outputdt = timedelta(hours=1)
    timerange = np.arange(np.nanmin(sim_data['time'].values),
                          np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                          outputdt)
    # dtsim = sim_data['time'].values[0][1]-sim_data['time'].values[0][0] # time-step in a single trajectory
    timewindow = (timerange[-2], timerange[-1])
    # We can only trully compute concentrations in a single time step equal to the output
    #    time_id = np.where(np.logical_and(sim_data['time'] >= timerange[0], sim_data['time'] <= timerange[-1]))  # Indices of the data where time = 0
    #    time_id = np.where(np.logical_and(sim_data['time'] >= timerange[-2],
    #                                      sim_data['time'] <= timerange[-1]))  # Indices of the data where time = 0
    # time_step = round((timewindow [1]-timewindow[0])/dtsim) # number of steps where accumulation is being computed
    time_id = np.where(np.logical_and(sim_data['time'] >= timewindow[0], sim_data['time'] < timewindow[1]))
    # time_id = np.where(np.logical_and(sim_data['beached']==0,np.logical_and(sim_data['time'] >= timewindow[0], sim_data['time'] < timewindow[1])))
    lon = sim_data['lon'].values[time_id]
    lat = sim_data['lat'].values[time_id]
    lon = lon.flatten()
    lon = lon[np.logical_not(np.isnan(lon))]
    lat = lat.flatten()
    lat = lat[np.logical_not(np.isnan(lat))]

    width_km = haversine(domain["W"], domain["N"], domain["E"], domain["N"])
    height_km = haversine(domain["W"], domain["N"], domain["W"], domain["S"])
    print(f'Map width={width_km:.3f} km')
    print(f'Map height={height_km:.3f} km')

    nbins = (round(1 * width_km), round(1 * height_km))
    # Equivalence of nbin in km2
    # binkm2 = width_km/nbins[0]*height_km/nbins[1]
    #    print(f'1 bin ={ binkm2:.3f} km2')
    # nbins=(round(abs(domain['W']-domain['E'])*factor_bins), round(abs(domain['S']-domain['N'])*factor_bins))
    print(f'Number of bins for 2d histogram: {nbins}')
    data, x_e, y_e = np.histogram2d(lon, lat, bins=nbins,
                                    range=[[domain['W'], domain['E']], [domain['S'], domain['N']]])
    # Tengo que hacer un promedio
    # data = data / time_step
    # data = data / binkm2
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([lon, lat]).T,
                method="splinef2d", bounds_error=False)
    idx = z.argsort()
    x, y, z = lon[idx], lat[idx], z[idx]
    scatter = ax.scatter(x, y, c=z)
    norm = Normalize(vmin=data.min(), vmax=data.max())
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel(f'Num particles per km$^2$')

    # t = np.datetime_as_string(timerange[0], unit='h')
    t = np.datetime_as_string(sim_data['time'].values[time_id][-1], unit='h')
    title = ax.set_title('Particles density at t = ' + t)
    ax.text(-0.1, 0.55, 'Latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    ax.text(0.5, -0.12, 'Longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize=12)
    plt.show()
    fig.savefig('heatmap.jpg')


def plot_hist2d_v2_video(sim_data, domain=None, terrain=None, terrain_zoom=10, show_plot=True, filename=None):
    land = True if terrain is None else False
    # Con fondo maritimo homogeneo
    plt, fig, ax, cartopy = create_parcelsfig_axis(True, land, projection=ccrs.PlateCarree(), cartopy_features=[])
    if domain is not None:
        new_domain = [domain['E'], domain['W'], domain['N'], domain['S']]
        ax.set_extent(new_domain, crs=cartopy.crs.PlateCarree())
    if terrain is not None:
        ax.add_image(terrain, terrain_zoom)
        # ax.coastlines()
        ax.stock_img()

    # set figure size
    fig.set_size_inches(13, 6)

    lon = np.ma.filled(sim_data.variables['lon'], np.nan)
    lat = np.ma.filled(sim_data.variables['lat'], np.nan)
    t = np.ma.filled(sim_data.variables['time'], np.nan)
    for p in range(lon.shape[1]):
        lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

    color_index = np.arange(lon.shape[0])

    # Para usar el mismo mapa de colores en plot que en scatter tengo que convertir el mapa a rgb
    phase = cmocean.cm.phase
    cNorm = colors.Normalize(vmin=0, vmax=color_index[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=phase)

    outputdt = timedelta(days=1)
    timerange = np.arange(np.nanmin(sim_data['time'].values),
                          np.nanmax(sim_data['time'].values) + np.timedelta64(outputdt),
                          outputdt)

    time_id = np.where(np.logical_and(sim_data['time'] >= timerange[0],
                                      sim_data['time'] <= timerange[1]))  # Indices of the data where time = 0
    lon = sim_data['lon'].values[time_id]
    lat = sim_data['lat'].values[time_id]
    lon = lon.flatten()
    lon = lon[np.logical_not(np.isnan(lon))]
    lat = lat.flatten()
    lat = lat[np.logical_not(np.isnan(lat))]

    width_km = haversine(domain["W"], domain["N"], domain["E"], domain["N"])
    height_km = haversine(domain["W"], domain["N"], domain["W"], domain["S"])
    print(f'Map width={width_km:.3f} km')
    print(f'Map height={height_km:.3f} km')
    nbins = (round(width_km), round(height_km))
    # nbins=(round(abs(domain['W']-domain['E'])*factor_bins), round(abs(domain['S']-domain['N'])*factor_bins))
    print(f'Number of bins for 2d histogram: {nbins}')
    data, x_e, y_e = np.histogram2d(lon, lat, bins=nbins,
                                    range=[[domain['W'], domain['E']], [domain['S'], domain['N']]])

    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([lon, lat]).T,
                method="splinef2d",
                bounds_error=False)
    idx = z.argsort()
    x, y, z = lon[idx], lat[idx], z[idx]
    scatter = ax.scatter(x, y, c=z)
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel(f'Num particles per km$^2$')

    t = np.datetime_as_string(timerange[0], unit='h')
    title = ax.set_title('Particles density at t = ' + t)

    def animate(i):
        t = np.datetime_as_string(timerange[i], unit='h')
        title.set_text('Particles density at t = ' + t)
        time_id = np.where(np.logical_and(sim_data['time'] >= timerange[i],
                                          sim_data['time'] <= timerange[i + 1]))  # Indices of the data where time = 0
        lon = sim_data['lon'].values[time_id]
        lat = sim_data['lat'].values[time_id]
        lon = lon.flatten()
        lon = lon[np.logical_not(np.isnan(lon))]
        lat = lat.flatten()
        lat = lat[np.logical_not(np.isnan(lat))]
        data, x_e, y_e = np.histogram2d(lon, lat, bins=nbins,
                                        range=[[domain['W'], domain['E']], [domain['S'], domain['N']]])

        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([lon, lat]).T,
                    method="splinef2d",
                    bounds_error=False)
        idx = z.argsort()
        x, y, z = lon[idx], lat[idx], z[idx]
        scatter.set_offsets(np.vstack([x, y]))
        # scatter.set_color(z)

    anim = FuncAnimation(fig, animate, frames=len(timerange) - 1, interval=500)
    if show_plot:
        plt.show()
    if filename is not None:
        logger.info(f"Saving animation video to {filename}")
        writervideo = FFMpegWriter(fps=30)
        anim.save(filename, writer=writervideo)
        logger.info("Video saved")

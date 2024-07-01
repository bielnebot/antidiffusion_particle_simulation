import pandas as pd
import numpy as np
import random
import math
import datetime
import os

import logging

logger = logging.getLogger(__name__)

from parcels import FieldSet, NestedField
from pathlib import Path
from parcels.plotting import create_parcelsfig_axis, cartopy_colorbar, parsetimestr
import cartopy.crs as ccrs
from parcels.field import Field, VectorField
from parcels.grid import GridCode, CurvilinearGrid
from parcels.tools.statuscodes import TimeExtrapolationError
import cmocean
import copy
import config

# def lat_lon_date(tracks_directory):
#     # input: directory where the .csv files are stored (str)
#     # output: list of the first lat, lon and date o each file

#     tracks_directory = tracks_directory if tracks_directory[-1] == "/" else tracks_directory + "/"

#     llistaFitxers = os.listdir(tracks_directory)

#     llistaData = []
#     llistaLat = []
#     llistaLon = []

#     for fitxer in llistaFitxers:
#         ruta = tracks_directory + fitxer
#         df = pd.read_csv(ruta, header=None)

#         dataInici, latInici, lonInici = df.iloc[0]
#         dayOfYear, _ = dataInici.split(" ")
#         Year, Month, Day = dayOfYear.split("-")

#         # if Month != "02":
#         #     continue

#         llistaData.append(np.datetime64(datetime(year=int(Year), month=int(Month), day=int(Day))))
#         llistaLat.append(latInici)
#         llistaLon.append(lonInici)

#     return llistaLat, llistaLon, llistaData

# def initial_coords_streamlit_read(file_object):
#     # input: streamlit file object
#     # output: first lat, lon and date coordinates of the file
#
#     df = pd.read_csv(file_object, header=None)
#
#     dataInici, latInici, lonInici = df.iloc[0]
#     dayOfYear, hourOfDay = dataInici.split(" ")
#     Year, Month, Day = dayOfYear.split("-")
#     Hour, Minute, Second = hourOfDay.split(":")
#
#     datetimeInici = datetime.datetime(year=int(Year), month=int(Month), day=int(Day), hour=int(Hour), minute=int(Minute),
#                  second=int(Second))
#     dateInici = datetime.date(year=int(Year), month=int(Month), day=int(Day))
#
#     return latInici, lonInici, datetimeInici, dateInici


def read_csv_file(file_object):
    """
    file_object can also be the uri
    """
    df = pd.read_csv(file_object, header=None)
    lat_coords = list(df[1])
    lon_coords = list(df[2])
    time_coords = list(df[0])
    time_coords = [datetime.datetime.strptime(data, '%Y-%m-%d %H:%M:%S') for data in time_coords]
    return lon_coords, lat_coords, time_coords

#
# def initialCoords_csv(tracks_directory):
#     # input: directory where the .csv file is stored (str)
#     # output: first lat, lon and date coordinates of the file
#
#     tracks_directory = tracks_directory if tracks_directory[-1] == "/" else tracks_directory + "/"
#
#     llistaFitxers = os.listdir(tracks_directory)
#
#     ruta = tracks_directory + llistaFitxers[0]
#     df = pd.read_csv(ruta, header=None)
#
#     dataInici, latInici, lonInici = df.iloc[0]
#     dayOfYear, hourOfDay = dataInici.split(" ")
#     Year, Month, Day = dayOfYear.split("-")
#     Hour, Minute, Second = hourOfDay.split(":")
#
#     datetimeInici = np.datetime64(
#         datetime.datetime(year=int(Year), month=int(Month), day=int(Day), hour=int(Hour), minute=int(Minute),
#                  second=int(Second)))
#
#     return latInici, lonInici, datetimeInici


def point_gauss_2d(mu, sigma):
    # input: mu = (muX, muY); sigma = (sigmaX, sigmaY)
    # output: a point in the 2D plane

    muX, muY = mu
    sigmaX, sigmaY = sigma
    x = random.normalvariate(muX, sigmaX)
    y = random.normalvariate(muY, sigmaY)
    return (x, y)


def point_cloud_gauss_2d(mu, sigma, amount):
    # input: mu = (muX, muY); sigma = (sigmaX, sigmaY)
    # output: a cloud of points
    return np.array([point_gauss_2d(mu, sigma) for _ in range(amount)])


def distance_to_angle(dist):
    # dist in km
    earth_radius = 6371  # [km]
    angle = dist / earth_radius
    return angle * 180 / math.pi


def point_cloud_from_coordinate(point, radius, amount, date):
    # point = [point_lat,point_lon]
    # radius in km. radius = 1 sigma of distributin (approx 68,3 % of points)
    degreesSigma = distance_to_angle(radius)
    print("degreesSigma=",degreesSigma)
    point_cloud = point_cloud_gauss_2d(point, [degreesSigma, degreesSigma], amount)

    llistaLat = point_cloud[:, 0]
    llistaLon = point_cloud[:, 1]

    if isinstance(date, list):
        llistaData = [np.datetime64(datetime.datetime(year=date[0], month=date[1], day=date[2])) for _ in range(amount)]
    elif isinstance(date, np.datetime64):
        llistaData = [date for _ in range(amount)]

    return llistaLat, llistaLon, llistaData


def get_ibi_fieldset():

    # Retrieve all current nc files
    currents_directory = Path(config.data_base_path) / Path(config.currents_files_directory)
    basepath_ibi = '*.nc'
    filenames_ibi = sorted(currents_directory.glob(str(basepath_ibi)))  # la llista de fitxers que hi ha a la carpeta

    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    U_ibi = Field.from_netcdf(filenames_ibi, ('U', 'uo'), dimensions, fieldtype='U', allow_time_extrapolation=False)
    V_ibi = Field.from_netcdf(filenames_ibi, ('V', 'vo'), dimensions, fieldtype='V', allow_time_extrapolation=False)

    return FieldSet(U_ibi, V_ibi)


def set_Stokes(fieldset, data_dir):

    fnames = Path(data_dir)
    print("fnames =",fnames)
    dimensionsU = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    dimensionsV = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    variablesU = ('Uuss', 'VSDX')
    variablesV = ('Vuss', 'VSDY')
    Uuss = Field.from_netcdf(fnames, variablesU, dimensionsU, fieldtype='U', allow_time_extrapolation=True)
    Vuss = Field.from_netcdf(fnames, variablesV, dimensionsV, fieldtype='V', allow_time_extrapolation=True,
                             grid=Uuss.grid, dataFiles=Uuss.dataFiles)
    fieldset.add_field(Uuss)
    fieldset.add_field(Vuss)
    fieldset.Uuss.vmax = 5
    fieldset.Vuss.vmax = 5
    uv_uss = VectorField('UVuss', fieldset.Uuss, fieldset.Vuss)
    fieldset.add_vector_field(uv_uss)


def set_Leeway(fieldset, data_dir, data_str):

    dir = Path(data_dir)
    print(dir)
    basepath = data_str
    fnames = sorted(dir.glob(str(basepath)))
    dimensionsU = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    dimensionsV = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    variablesU = ('Uuwind', 'u10')
    variablesV = ('Vuwind', 'v10')
    Uuwind = Field.from_netcdf(fnames, variablesU, dimensionsU, fieldtype='U', allow_time_extrapolation=True)
    Vuwind = Field.from_netcdf(fnames, variablesV, dimensionsV, fieldtype='V', allow_time_extrapolation=True,
                             grid=Uuwind.grid, dataFiles=Uuwind.dataFiles)
    fieldset.add_field(Uuwind)
    fieldset.add_field(Vuwind)
    # fieldset.Uuwind.vmax = 5
    # fieldset.Vuwind.vmax = 5
    uv_uwind = VectorField('UVwind', fieldset.Uuwind, fieldset.Vuwind)
    fieldset.add_vector_field(uv_uwind)


# def get_port_fieldset() -> FieldSet:
#     """
#     Carrega tots els fitxers de la carpeta de port i retorna el fieldset corresponent
#     """
#     dir = Path(cfg.data_base_path) / Path(cfg.harbour_files_dir)
#     filenames = {'U': sorted(dir.glob(cfg.port_file_search_string)),
#                  'V': sorted(dir.glob(cfg.port_file_search_string))}
#     variables = {'U': 'u', 'V': 'v'}
#     dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
#     fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
#     return fieldset
#
#
# def get_coastal_fieldset() -> FieldSet:
#     """
#     Carrega tots els fitxers de la carpeta de costa i retorna el fieldset corresponent
#     """
#     dir = Path(cfg.data_base_path) / Path(cfg.coastal_files_dir)
#     filenames = {'U': sorted(dir.glob(cfg.coastal_file_search_string)),
#                  'V': sorted(dir.glob(cfg.coastal_file_search_string))}
#     variables = {'U': 'u', 'V': 'v'}
#     dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
#     fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
#     return fieldset
#
#
# def get_mixed_fieldset() -> FieldSet:
#     """
#     Carrega tots els fitxers combinats de costa+port i retorna el fieldset corresponent
#     """
#     dir = Path(cfg.data_base_path) / Path(cfg.mixed_harbour_coastal_dir)
#     filenames = {'U': sorted(dir.glob("*.nc")),
#                  'V': sorted(dir.glob("*.nc"))}
#     variables = {'U': 'u', 'V': 'v'}
#     dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
#     fieldset = FieldSet.from_netcdf(filenames, variables, dimensions)
#     return fieldset
#
#
# def get_nested_fieldset() -> FieldSet:
#     """
#     Carga los ficheros de coastal + port domain y los anida
#     """
#     # Cargamos primero los ficheros domincio Portuario
#     dir_prt = Path(cfg.data_base_path) / Path(cfg.harbour_files_dir + '_resampled/')
#     basepath_prt = '*HC_resampled.nc'
#     filenames_prt = sorted(dir_prt.glob(str(basepath_prt)))
#
#     dimensions = {'time': 'time', 'depth': 'depth', 'lon': 'longitude', 'lat': 'latitude'}
#
#     U_prt = Field.from_netcdf(filenames_prt, ('U', 'u'), dimensions, fieldtype='U')
#     V_prt = Field.from_netcdf(filenames_prt, ('V', 'v'), dimensions, fieldtype='V')
#
#     # Cargamos los ficheros del dominio Costero
#     dir_cst = Path(cfg.data_base_path) / Path(cfg.coastal_files_dir)
#     basepath_cst = '*HC.nc'
#     filenames_cst = sorted(dir_cst.glob(str(basepath_cst)))
#
#     U_cst = Field.from_netcdf(filenames_cst, ('U', 'u'), dimensions, fieldtype='U')
#     V_cst = Field.from_netcdf(filenames_cst, ('V', 'v'), dimensions, fieldtype='V')
#
#     # Anidamos ahora los fieldsets
#     fieldset_prt = FieldSet(U_prt, V_prt)
#     # fieldset_prt.U.show()
#     fieldset_cst = FieldSet(U_cst, V_cst)
#
#     Ufield = NestedField('U', [fieldset_prt.U, fieldset_cst.U])
#     Vfield = NestedField('V', [fieldset_prt.V, fieldset_cst.V])
#     fieldset = FieldSet(Ufield, Vfield)
#
#     return fieldset
#
#
# def get_nested_fieldset_ibi() -> FieldSet:
#     """
#     Carga los ficheros de harbur-coastal, coastal-ibi, y IBI domain y los anida
#     """
#     base_path = Path(cfg.data_base_path)
#     # Cargamos primero los ficheros domincio Portuario
#     dir_prt = base_path / Path(cfg.harbour_files_dir + '_resampled/')
#     assert dir_prt.is_dir(), f"Directory not found: {dir_prt}"
#     basepath_prt = '*.nc'
#     filenames_prt = sorted(dir_prt.glob(str(basepath_prt)))
#     dimensions = {'time': 'time', 'depth': 'depth', 'lon': 'longitude', 'lat': 'latitude'}
#     U_prt = Field.from_netcdf(filenames_prt, ('U', 'u'), dimensions, fieldtype='U', allow_time_extrapolation=True)
#     V_prt = Field.from_netcdf(filenames_prt, ('V', 'v'), dimensions, fieldtype='V', allow_time_extrapolation=True)
#
#     # Cargamos los ficheros del dominio Costero
#     dir_cst = base_path / Path(cfg.coastal_files_dir + '_resampled/')
#     assert dir_cst.is_dir(), f"Directory not found: {dir_cst}"
#     basepath_cst = '*.nc'
#     filenames_cst = sorted(dir_cst.glob(str(basepath_cst)))
#     U_cst = Field.from_netcdf(filenames_cst, ('U', 'u'), dimensions, fieldtype='U', allow_time_extrapolation=True)
#     V_cst = Field.from_netcdf(filenames_cst, ('V', 'v'), dimensions, fieldtype='V', allow_time_extrapolation=True)
#
#     # Cargamos los ficheros del dominio Costero
#     dir_ibi = base_path / Path(cfg.currents_files_dir + '_resampled/')
#     assert dir_ibi.is_dir(), f"Directory not found: {dir_cst}"
#     basepath_ibi = '*.nc'
#     filenames_ibi = sorted(dir_ibi.glob(str(basepath_cst)))
#     U_ibi = Field.from_netcdf(filenames_ibi, ('U', 'u'), dimensions, fieldtype='U', allow_time_extrapolation=True)
#     V_ibi = Field.from_netcdf(filenames_ibi, ('V', 'v'), dimensions, fieldtype='V', allow_time_extrapolation=True)
#
#     # Anidamos ahora los fieldsets
#     fieldset_prt = FieldSet(U_prt, V_prt)
#     # fieldset_prt.U.show()
#     fieldset_cst = FieldSet(U_cst, V_cst)
#     fieldset_ibi = FieldSet(U_ibi, V_ibi)
#
#     Ufield = NestedField('U', [fieldset_prt.U, fieldset_cst.U, fieldset_ibi.U])
#     Vfield = NestedField('V', [fieldset_prt.V, fieldset_cst.V, fieldset_ibi.V])
#     fieldset = FieldSet(Ufield, Vfield)
#
#     return fieldset


def DeleteParticleFunction(particle, fieldset, time):
    particle.delete()


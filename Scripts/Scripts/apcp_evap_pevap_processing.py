#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:57:25 2020

@author: stuartedris

A script to process the data in the NARR files.

This assumes the user is in the MISC_Data/NARR/ directory
"""

#%%
# Load libraries

import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colorbar as mcolorbar
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from scipy import stats
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator

#%%
# A function to laod the data
def load3Dnc(filename, SName, path = '/Volumes/My Book/'):
    '''
    '''
    
    with Dataset(path + filename, 'r') as nc:
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
        
        time = nc.variables['time'][:]
        
        var = nc.variables[SName][:,:,:]
        
    return var, lat, lon, time

def load2Dnc(filename, SName, path = '/Volumes/My Book/'):
    '''
    '''
    
    with Dataset(path + filename, 'r') as nc:
        var = nc.variables[SName][:,:]
        
    return var

#%%
# Create a function to import the nc files
def LoadNC(filename, SName,  path = './Data/pentad_NARR_grid/'):
    '''
    
    '''
    
    X = {}
    DateFormat = '%Y-%m-%d %H:%M:%S' #Note here: If a file is being loaded that used NCWrite with datetimes as the date variable, %H:%M:%S is needed (this is the default format converted to by str()). 
                                     #  If NCWrite was used using a string list as the date variable, omit the %H:%M:%S as it is not part of the date format.
    
    with Dataset(path + filename, 'r') as nc:
        # Load the grid
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
#        lat = nc.variables['lat'][:]
#        lon = nc.variables['lon'][:]
#        
#        lon, lat = np.meshgrid(lon, lat)
#        
        X['lat'] = lat
        X['lon'] = lon
        
        # Collect the time information
        time = nc.variables['date'][:]
        dates = np.asarray([datetime.strptime(time[d], DateFormat) for d in range(len(time))])
        
        X['date'] = dates
        X['year']  = np.asarray([d.year for d in dates])
        X['month'] = np.asarray([d.month for d in dates])
        X['day']   = np.asarray([d.day for d in dates])
        X['ymd']   = np.asarray([datetime(d.year, d.month, d.day) for d in dates])
        
        # Load the mask data
        X['mask'] = nc.variables['mask'][:,:]
        
        # Collect the data itself
        X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X

# Function to load nc files without a land-sea mask is needed for the USDM data (which does not have a mask)
def LoadNCnomask(filename, SName,  path = './Data/pentad_NARR_grid/'):
    '''
    
    '''
    
    X = {}
    DateFormat = '%Y-%m-%d %H:%M:%S' #Note here: If a file is being loaded that used NCWrite with datetimes as the date variable, %H:%M:%S is needed (this is the default format converted to by str()). 
                                     #  If NCWrite was used using a string list as the date variable, omit the %H:%M:%S as it is not part of the date format.
    
    with Dataset(path + filename, 'r') as nc:
        # Load the grid
        lat = nc.variables['lat'][:,:]
        lon = nc.variables['lon'][:,:]
#        lat = nc.variables['lat'][:]
#        lon = nc.variables['lon'][:]
#        
#        lon, lat = np.meshgrid(lon, lat)
#        
        X['lat'] = lat
        X['lon'] = lon
        
        # Collect the time information
        time = nc.variables['date'][:]
        dates = np.asarray([datetime.strptime(time[d], DateFormat) for d in range(len(time))])
        
        X['date'] = dates
        X['year']  = np.asarray([d.year for d in dates])
        X['month'] = np.asarray([d.month for d in dates])
        X['day']   = np.asarray([d.day for d in dates])
        X['ymd']   = np.asarray([datetime(d.year, d.month, d.day) for d in dates])
    
        
        # Collect the data itself
        X[str(SName)] = nc.variables[str(SName)][:,:,:]
        
    return X

#%%
# A function to subset the data
def SubsetData(X, Lat, Lon, LatMin, LatMax, LonMin, LonMax):
    '''
    
    '''
    
    # Collect the original sizes of the data/lat/lon
    I, J, T = X.shape
    
    # Reshape the data into a 2D array and lat/lon to a 1D array for easier referencing.
    X2D   = X.reshape(I*J, T, order = 'F')
    Lat1D = Lat.reshape(I*J, order = 'F')
    Lon1D = Lon.reshape(I*J, order = 'F')
    
    # Find the indices in which to make the subset.
    LatInd = np.where( (Lat1D >= LatMin) & (Lat1D <= LatMax) )[0]
    LonInd = np.where( (Lon1D >= LonMin) & (Lon1D <= LonMax) )[0]
    
    # Find the points where the lat and lon subset overlap. This comprises the subsetted grid.
    SubInd = np.intersect1d(LatInd, LonInd)
    
    # Next find, the I and J dimensions of subsetted grid.
    Start = 0 # The starting point of the column counting.
    Count = 1 # Row count starts at 1
    Isub  = 0 # Start by assuming subsetted column size is 0.
    
    for n in range(len(SubInd[:-1])): # Exclude the last value to prevent indexing errors.
        IndDiff = SubInd[n+1] - SubInd[n] # Obtain difference between this index and the next.
        if (n+2) == len(SubInd): # At the last value, everything needs to be increased by 2 to account for the missing indice at the end.
            Isub = np.nanmax([Isub, n+2 - Start]) # Note since this is the last indice, and this row is counted, there is no Count += 1.
        elif ( (IndDiff > 1) |              # If the difference is greater than 1, or if
             (np.mod(SubInd[n]+1,I) == 0) ):# SubInd is divisible by I, then a new row 
                                            # is started in the gridded array.
            Isub = np.nanmax([Isub, n+1 - Start]) # Determine the highest column count (may not be the same from row to row)
            Start = n+1 # Start the counting anew.
            Count = Count + 1 # Increment the row count by 1 as the next row is entered.
        else:
            pass
        
    # At the end, Count has the total number of rows in the subset.
    Jsub = Count
    
    # Next, the column size may not be the same from row to row. The rows with
    # with columns less than Isub need to be filled in. 
    # Start by finding how many placeholders are needed.
    PH = Isub * Jsub - len(SubInd) # Total number of needed points - number in the subset
    
    # Initialize the variable that will hold the needed indices.
    PlaceHolder = np.ones((PH)) * np.nan
    
    # Fill the placeholder values with the indices needed to complete a Isub x Jsub matrix
    Start = 0
    m = 0
    
    for n in range(len(SubInd[:-1])):
        # Identify when row changes occur.
        IndDiff = SubInd[n+1] - SubInd[n]
        if (n+2) == len(SubInd): # For the end of last row, an n+2 is needed to account for the missing index (SubInd[:-1] was used)
            ColNum = n+2-Start
            PlaceHolder[m:m+Isub-ColNum] = SubInd[n+1] + np.arange(1, 1+Isub-ColNum)
            # Note this is the last value, so nothing else needs to be incremented up.
        elif ( (IndDiff > 1) | (np.mod(SubInd[n]+1,I) == 0) ):
            # Determine how man columns this row has.
            ColNum = n+1-Start
            
            # Fill the placeholder with the next index(ices) when the row has less than
            # the maximum number of columns (Isub)
            PlaceHolder[m:m+Isub-ColNum] = SubInd[n] + np.arange(1, 1+Isub-ColNum)
            
            # Increment the placeholder index by the number of entries filled.
            m = m + Isub - ColNum
            Start = n+1
            
        
        else:
            pass
    
    # Next, convert the placeholders to integer indices.
    PlaceHolderInt = PlaceHolder.astype(int)
    
    # Add and sort the placeholders to the indices.
    SubIndTotal = np.sort(np.concatenate((SubInd, PlaceHolderInt), axis = 0))
    
    # The placeholder indices are technically outside of the desired subset. So
    # turn those values to NaN so they do not effect calculations.
    # (In theory, X2D is not the same variable as X, so the original dataset 
    #  should remain untouched.)
    X2D[PlaceHolderInt,:] = np.nan
    
    # Collect the subset of the data, lat, and lon
    XSub = X2D[SubIndTotal,:]
    LatSub = Lat1D[SubIndTotal]
    LonSub = Lon1D[SubIndTotal]
    
    # Reorder the data back into a 3D array, and lat and lon into gridded 2D arrays
    XSub = XSub.reshape(Isub, Jsub, T, order = 'F')
    LatSub = LatSub.reshape(Isub, Jsub, order = 'F')
    LonSub = LonSub.reshape(Isub, Jsub, order = 'F')
    
    # Return the the subsetted data
    return XSub, LatSub, LonSub


#%%
# Calculate the climatological means and standard deviations
  
def CalculateClimatology(var, pentad = False):
    '''
    '''
    
    # Obtain the dimensions of the variable
    I, J, T = var.shape
    
    # Count the number of years
    if pentad is True:
        yearLen = int(365/5)
    else:
        yearLen = int(365)
        
    NumYear = int(np.ceil(T/yearLen))
    
    # Create a variable for each day, assumed starting at Jan 1 and no
    #   leap years (i.e., each year is only 365 days each)
    day = np.ones((T)) * np.nan
    
    n = 0
    for i in range(1, NumYear+1):
        if i >= NumYear:
            day[n:T+1] = np.arange(1, len(day[n:T+1])+1)
        else:
            day[n:n+yearLen] = np.arange(1, yearLen+1)
        
        n = n + yearLen
    
    # Initialize the climatological mean and standard deviation variables
    ClimMean = np.ones((I, J, yearLen)) * np.nan
    ClimStd  = np.ones((I, J, yearLen)) * np.nan
    
    # Calculate the mean and standard deviation for each day and at each grid
    #   point
    for i in range(1, yearLen+1):
        ind = np.where(i == day)[0]
        ClimMean[:,:,i-1] = np.nanmean(var[:,:,ind], axis = -1)
        ClimStd[:,:,i-1]  = np.nanstd(var[:,:,ind], axis = -1)
    
    return ClimMean, ClimStd


#%% 
# cell 5
# A function to create datetime datasets

def DateRange(StartDate, EndDate):
    '''
    This function takes in two dates and outputs all the dates inbetween
    those two dates.
    
    Inputs:
        StartDate - A datetime. The starting date of the interval.
        EndDate - A datetime. The ending date of the interval.
        
    Outputs:
        All dates between StartDate and EndDate (inclusive)
    '''
    for n in range(int((EndDate - StartDate).days) + 1):
        yield StartDate + timedelta(n) 

#%%
# Load in some data to examine and initialize other variables
path = './apcp/'
apcp, lat, lon, time = load3Dnc('apcp.1979.nc', SName = 'apcp', path = path)

# Fix the lon values
for i in range(len(lon[:,0])):
    ind = np.where( lon[i,:] > 0 )[0]
    lon[i,ind] = -1*lon[i,ind]
    
    
#%%
# Create a datetime dataset for all times
dates_gen = DateRange(datetime(1979, 1, 1), datetime(2019, 12, 31))
dates = np.asarray([date for date in dates_gen])

dates_gen = DateRange(datetime(1979, 1, 1), datetime(2019, 12, 31))
all_years = np.asarray([date.year for date in dates_gen])

# Initialize a full variable
years = np.arange(1979, 2020)

I, J = lat.shape
T = dates.size

#full_precip = np.ones((T, I, J)) * np.nan
#full_evap   = np.ones((T, I, J)) * np.nan
full_pevap  = np.ones((T, I, J)) * np.nan

#apcpPath = './apcp/'
#evapPath = './evap/'
pevapPath = './pevap/'

# Fill the variables
for year in years:
    ind = np.where(year == all_years)[0]
    apcpFN  = 'apcp.' + str(year) + '.nc'
    evapFN  = 'evap.' + str(year) + '.nc'
    pevapFN = 'pevap.' + str(year) + '.nc'
    
    # Load the variable for that year
    #apcp, lat, lon, time = load3Dnc(apcpFN, 'apcp', path = apcpPath)
    #evap, lat, lon, time = load3Dnc(evapFN, 'evap', path = evapPath)
    pevap, lat, lon, time = load3Dnc(pevapFN, 'pevap', path = pevapPath)
    
    # Add the data to the whole
    #full_precip[ind,:,:] = apcp[:,:,:]
    #full_evap[ind,:,:]   = evap[:,:,:]
    full_pevap[ind,:,:]  = pevap[:,:,:]


#%%
# Delete leap years and convert to a lat x lon x time array
    
# Since this deletes part of the array, make sure to only run this once per script run.
T, I, J = full_pevap.shape
print('1')
dates_gen = DateRange(datetime(1979, 1, 1), datetime(2019, 12, 31))
all_days = np.asarray([date.day for date in dates_gen])

print('2')
dates_gen = DateRange(datetime(1979, 1, 1), datetime(2019, 12, 31))
all_months = np.asarray([date.month for date in dates_gen])

print('3')
ind = np.where( (all_months == 2) & (all_days == 29) )[0]

Tnew = T - len(ind)
print('4')
#precip_full = np.ones((I, J, Tnew)) * np.nan
#evap_full   = np.ones((I, J, Tnew)) * np.nan
pevap_full  = np.ones((I, J, Tnew)) * np.nan

print('5')
n = 0
for t in range(T):
    print(np.round(t/T*100,2))
    if (dates[t].month == 2) & (dates[t].day == 29):
        n = n + 1
        continue
    else:
        tn = t - n
        #precip_full[:,:,tn] = full_precip[t,:,:]
        #evap_full[:,:,tn]   = full_evap[t,:,:]
        pevap_full[:,:,tn]  = full_pevap[t,:,:]


#%%
# Load a mask and subset datasets
mask = load2Dnc('land.nc', 'land', path = './')

# Turn mask from time x lat x lon into lat x lon x time
T, I, J = mask.shape

maskNew = np.ones((I, J, T)) * np.nan
maskNew[:,:,0] = mask[0,:,:] # No loop is needed since the time dimension has length 1

LatMin = 25
LatMax = 50
LonMin = -130
LonMax = -65

#precip_sub, LatSub, LonSub = SubsetData(precip_full, lat, lon, LatMin, LatMax, LonMin, LonMax)
#evap_sub, LatSub, LonSub   = SubsetData(evap_full, lat, lon, LatMin, LatMax, LonMin, LonMax)
pevap_sub, LatSub, LonSub  = SubsetData(pevap_full, lat, lon, LatMin, LatMax, LonMin, LonMax)
mask_sub, LatSub, LonSub   = SubsetData(maskNew, lat, lon, LatMin, LatMax, LonMin, LonMax)

#%%
# Apply a land-sea mask
I, J, Tnew = pevap_sub.shape

print('Reshaping arrays')
mask_sub = mask_sub.reshape(I*J, 1, order = 'F')
#precip_sub = precip_sub.reshape(I*J, Tnew, order = 'F')
#evap_sub   = evap_sub.reshape(I*J, Tnew, order = 'F')
pevap_sub  = pevap_sub.reshape(I*J, Tnew, order = 'F')

print('Applying mask')
ind = np.where(mask_sub[0,:] == 0)[0] # 0 = sea, 1 = land
#precip_sub[ind,:] = np.nan
#evap_sub[ind,:]   = np.nan
pevap_sub[ind,:]  = np.nan

print('Reshaping arrays back')
mask_sub = mask_sub.reshape(I, J, 1, order = 'F')
#precip_sub = precip_sub.reshape(I, J, Tnew, order = 'F')
#evap_sub   = evap_sub.reshape(I, J, Tnew, order = 'F')
pevap_sub  = pevap_sub.reshape(I, J, Tnew, order = 'F')

#%%
# Write the full years of data.

def WriteNC(var, lat, lon, dates, mask, filename = 'tmp.nc', VarName = 'tmp', path = './'):
    '''
    '''
    
    # Determine the spatial and temporal lengths
    J ,I, T = var.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = 'This file contains data from the NCEP North American ' +\
                         'Regional Reanalysis model for all years from 1979 to ' +\
                         '2019 for a given variable and subsetted over CONUS. This file contains ' + str(VarName) +\
                         '. This data is at the daily time scale.\n' +\
                         'Variable: ' + str(VarName) + ' (kg m^-2, or unitless for ratios and indexes). This is the ' +\
                         'main variable for this file. It is in the format ' +\
                         'lon x lat x time.\n' +\
                         'lat: The latitude (matrix form).\n' +\
                         'lon: The longitude (matrix form).\n' +\
                         'date: List of dates starting from ' +\
                         '01-01-1979 to 12-31-2019 (%Y-%m-%d format). Leap year additions are excluded.' +\
                         'mask: Land-sea mask. 0 = sea, 1 = land.'
        
        # Create the spatial and temporal dimensions
        nc.createDimension('x', size = J)
        nc.createDimension('y', size = I)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables
        nc.createVariable('lat', lat.dtype, ('x', 'y'))
        nc.createVariable('lon', lon.dtype, ('x', 'y'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = np.str(dates[n])
            
        # Create the mask
        nc.createVariable('mask', int, ('x', 'y'))
        nc.variables['mask'] = mask[:,:]
            
        # Create the main variable
        nc.createVariable(VarName, var.dtype, ('x', 'y', 'time'))
        nc.variables[str(VarName)][:,:,:] = var[:,:,:]   

# Write a dates variable that excludes leap years
dates_new = ['tmp']*Tnew
n = 0
for t, date in enumerate(dates):
    if (date.month == 2) & (date.day == 29):
        n = n + 1
        continue
    else:
        tn = t - n
        dates_new[tn] = date.strftime('%Y-%m-%d')


#WriteNC(precip_sub, LatSub, LonSub, dates_new, mask_sub[:,:,0], filename = 'apcp_all_years_conus.nc', VarName = 'apcp', path = apcpPath)
#WriteNC(evap_sub, LatSub, LonSub, dates_new, mask_sub[:,:,0], filename = 'evap_all_years_conus.nc', VarName = 'evap', path = evapPath)
WriteNC(pevap_sub, LatSub, LonSub, dates_new, mask_sub[:,:,0], filename = 'pevap_all_years_conus.nc', VarName = 'pevap', path = pevapPath)

#%%
# Redefine the nc write function so that its description better describes the indices
def WriteNC(var, lat, lon, dates, mask, filename = 'tmp.nc', VarName = 'tmp', path = './'):
    '''
    '''
    
    # Determine the spatial and temporal lengths
    J ,I, T = var.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = 'This file contains data from the NCEP North American ' +\
                         'Regional Reanalysis model for all years from 1979 to ' +\
                         '2019 for a given index and subsetted over CONUS. This file contains ' + str(VarName) +\
                         '. This data is at the daily time scale.\n' +\
                         'Variable: ' + str(VarName) + ' (unitless). This is the ' +\
                         'main variable for this file. It is in the format ' +\
                         'lon x lat x time.\n' +\
                         'lat: The latitude (matrix form).\n' +\
                         'lon: The longitude (matrix form).\n' +\
                         'date: List of dates starting from ' +\
                         '01-01-1979 to 12-31-2019 (%Y-%m-%d format). Leap year additions are excluded.' +\
                         'mask: Land-sea mask. 0 = sea, 1 = land.'
        
        # Create the spatial and temporal dimensions
        nc.createDimension('x', size = J)
        nc.createDimension('y', size = I)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables
        nc.createVariable('lat', lat.dtype, ('x', 'y'))
        nc.createVariable('lon', lon.dtype, ('x', 'y'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = np.str(dates[n])
            
        # Create the mask
        nc.createVariable('mask', int, ('x', 'y'))
        nc.variables['mask'] = mask[:,:]
            
        # Create the main variable
        nc.createVariable(VarName, var.dtype, ('x', 'y', 'time'))
        nc.variables[str(VarName)][:,:,:] = var[:,:,:] 

#%%
# Calculate and write the ESR file
        
# If the computer is running slowly, reset/nuke/clear everything to free up memory, rerun the necessary functions and run the following 2 lines
ET = LoadNC('evap_all_years_conus.nc', 'evap', path = './evap/')
PET = LoadNC('pevap_all_years_conus.nc', 'pevap', path = './pevap/')


esr = ET['evap']/PET['pevap']
WriteNC(esr, ET['lat'], ET['lon'], ET['ymd'], ET['mask'], filename = 'esr_all_years_conus.nc', VarName = 'esr', path = './esr/')

#%%
# Calculate and write SESR for all points and time

# Calculate the climatological mean and standard deviations
print('Calculating Climatology')
esr_mean, esr_std = CalculateClimatology(esr, pentad = False)

# Initialize SESR and a new date variable
print('Initializing variables')
I, J, T = esr.shape

sesr = np.ones((I, J, T)) * np.nan

single_year_gen = DateRange(datetime(2010, 1, 1), datetime(2010, 12, 31)) # The choice of year here is arbitrary. It jsut has to not be a leap year.
single_year = np.asarray([date for date in single_year_gen])


# Calculate SESR
print('Calculating SESR')
for t, date in enumerate(single_year):
    print(round(t/len(single_year)*100, 2))
    ind = np.where( (date.month == ET['month']) & (date.day == ET['day']) )[0] # Find the point in all years corresponding to this date
    
    for day in ind:
        sesr[:,:,day] = (esr[:,:,day] - esr_mean[:,:,t])/esr_std[:,:,t]
    
print('Reshaping SESR back and writing the file')
# sesr = sesr.reshape(I, J, T, reorder = 'F')

# Write SESR to a seperate file
WriteNC(sesr, ET['lat'], ET['lon'], ET['ymd'], ET['mask'], filename = 'sesr_all_years_conus.nc', VarName = 'sesr', path = './sesr/')


#%%
# Calculate and write the drought percentiles for all points and years

# If the computer is running slowly, reset/nuke/clear everything to free up memory, rerun the necessary functions and run the following line
sesr = LoadNC('sesr_all_years_conus.nc', 'sesr', path = './sesr/')

# Initialize the sesr percentiles
print('Initializing and reshaping')
I, J, T = sesr['sesr'].shape

sesr_percentiles = np.ones((I, J, T)) * np.nan

# Reshape everything
sesr2d = sesr['sesr'].reshape(I*J, T, order = 'F')
sesr_percentiles = sesr_percentiles.reshape(I*J, T, order = 'F')

# Calculate the percentiles of SESR
print('Calculate the percentiles of SESR')
for t in range(T):
    if np.mod(t, 100) == 0:
        print(np.round(t/T*100, 2))
    
    ind = np.where( (sesr['ymd'][t].month == sesr['month']) & (sesr['ymd'][t].day == sesr['day']) )[0] # Find all indices corresponding to the current date
    for ij in range(I*J): # Annoyingly, percentile of score does not output arrays, so each grid point has to be looped over individually.
        sesr_percentiles[ij,t] = stats.percentileofscore(sesr2d[ij,ind], sesr2d[ij,t])
        
print('Reshaping SESR percentiles back and writing the file')


# Write SESR to a seperate file
sesr_percentiles = sesr_percentiles.reshape(I, J, T, order = 'F')

WriteNC(sesr_percentiles, sesr['lat'], sesr['lon'], sesr['ymd'], sesr['mask'], filename = 'sesr_percentiles_all_years_daily_conus.nc', VarName = 'SP', path = '../Drought_Data/')


#%%
# Next, calulcate and write the SPI for all points and time.

# If the computer is running slowly, reset/nuke/clear everything to free up memory, rerun the necessary functions and run the following line
P = LoadNC('apcp_all_years_conus.nc', 'apcp', path = './apcp/')

# Initialize SPI and a new date variable
print('Initializing variables')
I, J, T = P['apcp'].shape

SPI = np.ones((I, J, T)) * np.nan

single_year_gen = DateRange(datetime(2010, 1, 1), datetime(2010, 12, 31)) # The choice of year here is arbitrary. It jsut has to not be a leap year.
single_year = np.asarray([date for date in single_year_gen])

# Reshape arrays
print('Reshaping arrays')
P2d = P['apcp'].reshape(I*J, T, order = 'F')
SPI = SPI.reshape(I*J, T, order = 'F')

# Calculate SPI
print('Calculating SPI')
for t, date in enumerate(single_year):
    print(round(t/len(single_year)*100, 2))
    ind = np.where( (date.month == P['month']) & (date.day == P['day']) )[0] # Find the point in all years corresponding to this date
    
    # Precipitation is dsitribution according to a gamma distribution. Find the parameters of the gamme distribution
    alpha = np.nanstd(P2d[:,ind], axis = -1)**2/(np.nanmean(P2d[:,ind], axis = -1)**2)
    beta  = np.nanmean(P2d[:,ind], axis = -1)/alpha
    
    P_mean = alpha * beta
    P_std  = alpha * (beta**2)
    
    for ij in range(I*J):
        # Transform the values into a normal distribution with mean 0 and standard deviation 1 using the inverse CDF method.
        # That is, if P has a gamma distribution, then cdf(P) is a uniform distribution. Then cdf_n^-1(cdf(P)) is normally distributed. cdf_n^-1 is the inverse of the normal cdf
        
        # the ppf, percent point function, is the inverse cdf. Its default arguements are mean (loc) = 0, and std (scale) = 1
        SPI[ij,ind] = stats.norm.ppf(stats.gamma.cdf(P2d[ij,ind], a = alpha[ij], loc = P_mean[ij], scale = P_std[ij]))
        
# Shape SPI back and write it
print('Reshaping SPI back and writing the file')
SPI = SPI.reshape(I, J, T, order = 'F')

SPI = np.where( (SPI != np.inf) & (SPI != -1*np.inf), SPI, 0) # Remove inf values (may occur where the mean is 0)

WriteNC(SPI, P['lat'], P['lon'], P['ymd'], P['mask'], filename = 'spi_all_years_daily_conus.nc', VarName = 'SPI', path = '../Drought_Data/')

#%%
# Recreate the NC writing function with a description for indices on USDM time scale
def WriteNC(var, lat, lon, dates, mask, filename = 'tmp.nc', VarName = 'tmp', path = './'):
    '''
    '''
    
    # Determine the spatial and temporal lengths
    J ,I, T = var.shape
    T = len(dates)
    
    with Dataset(path + filename, 'w', format = 'NETCDF4') as nc:
        # Write a description for the .nc file
        nc.description = 'This file contains data from the NCEP North American ' +\
                         'Regional Reanalysis model for all years from 1979 to ' +\
                         '2019 for a given index and subsetted over CONUS. This file contains ' + str(VarName) +\
                         '. This data is at the weekly time scale, with dates corresponding to end of ' +\
                         'time interval and dates corresponding to the USDM data. \n' +\
                         'Variable: ' + str(VarName) + ' (unitless). This is the ' +\
                         'main variable for this file. It is in the format ' +\
                         'lon x lat x time.\n' +\
                         'lat: The latitude (matrix form).\n' +\
                         'lon: The longitude (matrix form).\n' +\
                         'date: List of dates starting from ' +\
                         '01-01-2010 to 12-31-2019 (%Y-%m-%d format). Leap year additions are excluded.' +\
                         'mask: Land-sea mask. 0 = sea, 1 = land.'
        
        # Create the spatial and temporal dimensions
        nc.createDimension('x', size = J)
        nc.createDimension('y', size = I)
        nc.createDimension('time', size = T)
        
        # Create the lat and lon variables
        nc.createVariable('lat', lat.dtype, ('x', 'y'))
        nc.createVariable('lon', lon.dtype, ('x', 'y'))
        
        nc.variables['lat'][:,:] = lat[:,:]
        nc.variables['lon'][:,:] = lon[:,:]
        
        # Create the date variable
        nc.createVariable('date', str, ('time', ))
        for n in range(len(dates)):
            nc.variables['date'][n] = np.str(dates[n])
            
        # Create the mask
        nc.createVariable('mask', int, ('x', 'y'))
        nc.variables['mask'] = mask[:,:]
            
        # Create the main variable
        nc.createVariable(VarName, var.dtype, ('x', 'y', 'time'))
        nc.variables[str(VarName)][:,:,:] = var[:,:,:] 

#%%
# Collect the SPI and drought percentiles for 2010 - 2019 and reduce associated variables to USDM time scales. Write them at the end.

# If the computer is running slowly, reset/nuke/clear everything to free up memory, rerun the necessary functions and run the following 3 lines
sesr = LoadNC('sesr_all_years_conus.nc', 'sesr', path = './sesr/')
SP = LoadNC('sesr_percentiles_all_years_daily_conus.nc', 'SP', path = '../Drought_Data/')
SPI = LoadNC('spi_all_years_daily_conus.nc', 'SPI', path = '../Drought_Data/')

# Load in USDM data
usdm = LoadNCnomask('USDM_grid_all_years.nc', 'USDM', path = '../../USDM_Data_Collection/USDM_Data/')

# Initialize variables
I, J, T = usdm['USDM'].shape

sesr_usdm = np.ones((I, J, T)) * np.nan
sp_usdm   = np.ones((I, J, T)) * np.nan
spi_usdm  = np.ones((I, J, T)) * np.nan

# Average all variables to the weekly time scales of the USDM
for t, date in enumerate(usdm['ymd']):
    ind = np.where( (sesr['ymd'] >= (date - timedelta(days = 6))) & (sesr['ymd'] <= date) )[0] # Find all days within the current week in the USDM dataset. -6 days because the current day is included.
    
    sesr_usdm[:,:,t] = np.mean(sesr['sesr'][:,:,ind], axis = -1)
    sp_usdm[:,:,t]   = np.mean(SP['SP'][:,:,ind], axis = -1)
    spi_usdm[:,:,t]  = np.mean(SPI['SPI'][:,:,ind], axis = -1)

# Next, write the variables.

WriteNC(sesr_usdm, usdm['lat'], usdm['lon'], usdm['ymd'], sesr['mask'], filename = 'sesr_all_years_USDMTimeScale_conus.nc', VarName = 'sesr', path = '../Drought_Data/')
WriteNC(sp_usdm, usdm['lat'], usdm['lon'], usdm['ymd'], sesr['mask'], filename = 'sesrPercentiles_all_years_USDMTimeScale_conus.nc', VarName = 'SP', path = '../Drought_Data/')
WriteNC(spi_usdm, usdm['lat'], usdm['lon'], usdm['ymd'], sesr['mask'], filename = 'SPI_all_years_USDMTimeScale_conus.nc', VarName = 'SPI', path = '../Drought_Data/')


#%%
# Make two quick plots to ensure the data is correct.
Year= 2012
Month = 6

ind = np.where( (usdm['year'] == Year) & (usdm['month'] == 6) )[0]

# SESR percentiles colorbar information
cmin_p = 0; cmax_p = 20; cint_p = 0.5
clevs_p = np.arange(cmin_p, cmax_p, cint_p)
nlevs_p = len(clevs_p)
cmap_p = plt.get_cmap(name = 'hot', lut = nlevs_p)

# SPI colorbar information
cmin = -3; cmax = 3; cint = 0.1
clevs = np.arange(cmin, cmax+cint, cint)
nlevs = len(clevs)
cmap  = plt.get_cmap(name = 'RdBu', lut = nlevs)

# Lat/Lon information
lat_int = 10
lon_int = 20

LatLabel = np.arange(-90, 90, lat_int)
LonLabel = np.arange(-180, 180, lon_int)

LonFormatter = cticker.LongitudeFormatter()
LatFormatter = cticker.LatitudeFormatter()

# Map projection information
fig_proj  = ccrs.PlateCarree()
data_proj = ccrs.PlateCarree()

# Additional shapefiles for removing non-US countries
ShapeName = 'admin_0_countries'
CountriesSHP = shpreader.natural_earth(resolution = '110m', category = 'cultural', name = ShapeName)

CountriesReader = shpreader.Reader(CountriesSHP)

USGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] == 'United States of America']
NonUSGeom = [country.geometry for country in CountriesReader.records() if country.attributes['NAME'] != 'United States of America']

# SESR Percentiles map
fig = plt.figure(figsize = [16, 18], frameon = True)
fig.suptitle('Average SESR Percentiles for June 2012', y = 0.68, size = 20)

# Set the first part of the figure
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)


ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
ax.add_feature(cfeature.STATES)
ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)

ax.set_xticks(LonLabel, crs = fig_proj)
ax.set_yticks(LatLabel, crs = fig_proj)

ax.set_yticklabels(LatLabel, fontsize = 18)
ax.set_xticklabels(LonLabel, fontsize = 18)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

cs = ax.pcolormesh(usdm['lon'], usdm['lat'], np.nanmean(sp_usdm[:,:,ind], axis = -1), vmin = cmin_p, vmax = cmax_p,
                 cmap = cmap_p, transform = data_proj, zorder = 1)

ax.set_extent([-129, -65, 25-1.5, 50-1.5])
cbax = fig.add_axes([0.12, 0.29, 0.78, 0.015])
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

for i in cbar.ax.get_xticklabels():
    i.set_size(18)

plt.show(block = False)




# SPI map
fig = plt.figure(figsize = [16, 18], frameon = True)
fig.suptitle('Average SPI for June 2012', y = 0.68, size = 20)

# Set the first part of the figure
ax = fig.add_subplot(1, 1, 1, projection = fig_proj)


ax.add_feature(cfeature.OCEAN, facecolor = 'white', edgecolor = 'none', zorder = 2)
ax.add_feature(cfeature.STATES)
ax.add_geometries(USGeom, crs = fig_proj, facecolor = 'none', edgecolor = 'black', zorder = 2)
ax.add_geometries(NonUSGeom, crs = fig_proj, facecolor = 'white', edgecolor = 'none', zorder = 2)

ax.set_xticks(LonLabel, crs = fig_proj)
ax.set_yticks(LatLabel, crs = fig_proj)

ax.set_yticklabels(LatLabel, fontsize = 18)
ax.set_xticklabels(LonLabel, fontsize = 18)

ax.xaxis.set_major_formatter(LonFormatter)
ax.yaxis.set_major_formatter(LatFormatter)

cs = ax.pcolormesh(usdm['lon'], usdm['lat'], np.nanmean(spi_usdm[:,:,ind], axis = -1), vmin = cmin, vmax = cmax,
                 cmap = cmap, transform = data_proj, zorder = 1)

ax.set_extent([-129, -65, 25-1.5, 50-1.5])
cbax = fig.add_axes([0.12, 0.29, 0.78, 0.015])
cbar = fig.colorbar(cs, cax = cbax, orientation = 'horizontal')

for i in cbar.ax.get_xticklabels():
    i.set_size(18)

plt.show(block = False)

# Notes:
#  SESR percentiles seem to be good compare with previous study.
#  SPI values seem much diminished. Figure 3 in Basara et al. 2019, SPI ranges from 2 to -3 in June, 2012.
#  The the figure made here, SPI seems to range from 1 to -1, This might be an error in the calculations, or it might simply be that the
#  SPI was calculated on the daily scale as opposed to the normal monthly time scale (the values have some variation depending on the average time scale).
#  This lower SPI values should potentially be kept in mind.

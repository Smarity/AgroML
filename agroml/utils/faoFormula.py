#%%
import math 

import ephem

import pandas as pd
import numpy as np

from icecream import ic

# Basic mathematical operations
pi = math.pi 
sin = math.sin
cos = math.cos
acos = math.acos
tan = math.tan
exp = math.exp
sqrt = math.sqrt


#  It calculates the relative distance from Earth to the sun
def getRelativeDistance(JulianDay):
    '''
    Input:
        JulianDay {int}
            Julian day
    Output:
        dr {float}
            Relative distance from Earth to the sun in meters
    '''
    dr = 1 + 0.033*cos(2*pi*JulianDay/365)
    return float(dr)
    
# It calculates the declination of the sun
def getSolarDeclination(JulianDay):
    '''
    Input:
        JulianDay {int}
        
    Output:
        delta {float}
            Declination of the sun in radians
    '''
    delta = 0.409*sin(2*pi*JulianDay/365 - 1.39)
    return float(delta)

# It calculates the sunset hour angle
def getSunsetHourAngle(JulianDay, Latitude):
    '''
    Input:
        JulianDay {int}
            Julian day
        Latitude {float}
            Latitude of the location in radians
    Output:
        Ws {float}
            Sunset hour angle in radians
    '''
    #Calculation of the solar declination (delta) [rad]
    delta=getSolarDeclination(JulianDay)

    Ws = acos(-tan(Latitude)*tan(delta))
    return float(Ws)

# It calculates of the extraterrestrial solar radiation
def getExtraterrestrialSolarRadiation(JulianDay, Latitude):
    '''
    
    Input:
        JulianDay {int}
            Julian day
        Latitude {float}
            Latitude of the location in radians
    Output:
        Ra {float}
            Extraterrestial solar radiation in MJ/(m**2 * day)
    '''
    dr = getRelativeDistance(JulianDay) # dr = relative distance
    Ws = getSunsetHourAngle(JulianDay,Latitude) # Ws = sunset hour angle
    delta = getSolarDeclination(JulianDay) # delta = solar declination
    phi = Latitude
    Gsc = 0.082 #solar constant [MJ/(m**2 * min)]

    Ra = (24*60/pi) * Gsc * dr * ((Ws * sin(phi) * sin(delta)) + (cos(phi) * cos(delta) * sin(Ws)))
    return float(Ra)

# It estimates the atmospheric pressure according to the elevation
def getAtmosphericPressure(altitude):
    '''
    Input:
        elevation {float}
            Site elevation in meters
    Output:
        pressure {float} 
            Atmospheric pressure in kPa
    '''
    pressure = 101.3 * ((293-0.0065*altitude)/293)**5.26
    return float(pressure)


def Calc_SaturationSlope(Tmax, Tmin):
    '''
    Calculation of the slope of the saturation vapour pressure [kPa/ºC]

    Input:
        Tmax [ºC] - 'float'
        Tmin [ºC] - 'float'
    Output:
        delta [kPa / ºC] - 'float'
    '''
    Tc = 0.5*(Tmax+Tmin)
    delta = 2504 * exp((17.27*Tc)/(Tc+237.2)) / (Tc + 237.3)**2
    return float(delta)

# %%

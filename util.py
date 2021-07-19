import numpy as np
import pandas as pd


def polar_warp(X, r, theta):
    return np.array([X[:, 0] + r * np.cos(theta), X[:, 1] + r * np.sin(theta)]).T


def get_st_coordinates(df):
    """
    Extracts spatial coordinates from ST data with index in 'AxB' type format.

    Return: pandas dataframe of coordinates
    """
    coor = []
    for spot in df.index:
        coordinates = spot.split("x")
        coordinates = [float(i) for i in coordinates]
        coor.append(coordinates)
    return np.array(coor)


def compute_distance(X1, X2):
    return np.sum(np.sqrt(np.sum((X1 - X2) ** 2, axis=1)))

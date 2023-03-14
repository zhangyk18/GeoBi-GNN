import numpy as np
import numba
from numba import njit, prange
import hausdorff.distances as distances
from inspect import getmembers

numba.config.NUMBA_DEFAULT_NUM_THREADS = 4


def _find_available_functions(module_name):
    all_members = getmembers(module_name)
    available_functions = [member[0] for member in all_members
                           if isinstance(member[1], numba.core.registry.CPUDispatcher)]
    return available_functions


@njit(fastmath=True, parallel=True)
def _nearest_distance(XA, XB, distance_function):
    ret = np.zeros(XA.shape[0], dtype=XA.dtype)
    nA = XA.shape[0]
    nB = XB.shape[0]
    for i in prange(nA):
        cmin = np.inf
        for j in prange(nB):
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
        ret[i] = cmin
    return ret


def nearest_distance(XA, XB, distance='euclidean'):
    assert type(XA) is np.ndarray and type(XB) is np.ndarray, 'arrays must be of type numpy.ndarray'
    assert np.issubdtype(XA.dtype, np.number) and np.issubdtype(XA.dtype, np.number), 'the arrays data type must be numeric'
    assert XA.ndim == 2 and XB.ndim == 2, 'arrays must be 2-dimensional'
    assert XA.shape[1] == XB.shape[1], 'arrays must have equal number of columns'

    if isinstance(distance, str):
        assert distance in _find_available_functions(distances), 'distance is not an implemented function'
        if distance == 'haversine':
            assert XA.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
            assert XB.shape[1] >= 2, 'haversine distance requires at least 2 coordinates per point (lat, lng)'
        distance_function = getattr(distances, distance)
    elif callable(distance):
        distance_function = distance
    else:
        raise ValueError("Invalid input value for 'distance' parameter.")

    return _nearest_distance(XA, XB, distance_function)


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def apply (coordinates:np.ndarray, affine:np.ndarray):

    n_coordinates = coordinates.shape[0]
    n_dimensions = coordinates.shape[1]

    assert(affine.shape[0] == n_dimensions + 1 and affine.shape[1] == n_dimensions + 1)

    coordinates_expanded = np.append(coordinates, np.ones((n_coordinates, 1)), axis=1)
    coordinates_transformed = (affine.astype(float) @ coordinates_expanded.T).T

    return coordinates_transformed[:,:3]

def apply_inverse (coordinates:np.ndarray, affine:np.ndarray):

    h = affine.shape[0] - 1
    w = affine.shape[1] - 1

    M = affine[:h, :w]
    b = affine[:h, w]

    invM = np.linalg.inv(M)
    invb = -invM.dot(b)

    inverse_affine = np.copy(affine.astype(float))
    inverse_affine[:h, :w] = invM
    inverse_affine[:h, w] = invb

    return apply(coordinates, inverse_affine)
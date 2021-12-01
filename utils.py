import numpy as np
from numpy import linalg as lalg
from .constants import C


def gamma_factor(velocity: float) -> float:
    """
    Calculates the Lorentz factor for a given velocity.
    Expects velocity as a multiple of c, i.e. velocity_in_mps = c * velocity.
    """
    return 1 / np.sqrt(1 - (velocity ** 2))


def lorentz_transformation(coords: np.ndarray, velocity: float) -> np.ndarray:
    """
    Computes the Lorentz transformation for a given time, position, and velocity.
    The coordinates are of shape (N, 2), where the coordinates are (t, x).
    The velocity is the relative velocity of the new frame relative to this frame.
    Expects velocity as a multiple of c, i.e. velocity_in_mps = c * velocity.
    """
    gamma = gamma_factor(velocity)
    T = gamma * np.array([[1, -velocity / C], [-C * velocity, 1]])
    return coords @ T.T


def velocity_transformation(frame_velocity: float, observed_velocity: float) -> float:
    """
    Computes the velocity transformation for a given frame velocity and observed velocity.
    The frame velocity is the velocity of the new frame relative to this frame.
    The observed velocity is a velocity measured in this frame.
    The returned velocity is the velocity of the observed velocity in the new frame.
    Expects velocity as a multiple of c, i.e. velocity_in_mps = c * velocity.
    """
    return (observed_velocity - frame_velocity) / (
        1 - frame_velocity * observed_velocity
    )

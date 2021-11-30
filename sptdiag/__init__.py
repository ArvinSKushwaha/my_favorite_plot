from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as lalg

# Constants
C = 1


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


class InertialFrame:
    def __init__(self, relative_velocity: float):
        """
        Constructs a new frame with a given velocity relative to the inertial frame.
        Expects relative velocity as a multiple of c, 
        i.e. relative_velocity_in_mps = c * relative_velocity.
        """
        self.relative_velocity = relative_velocity  # Velocity relative to the observer

        # Collection of data with shape (N, 2) with coordinates (t, x)
        self.data: List[np.ndarray] = []

    def add(self, data: np.ndarray) -> None:
        """
        Adds data in the form of 2D numpy arrays to the spacetime diagram.
        """
        self.data.append(data)

    def to_observer(self) -> "InertialFrame":
        """
        Transforms the spacetime diagram data to the observer frame.
        """
        frame = InertialFrame(0)
        for data in self.data:
            transformed_data = lorentz_transformation(data, -self.relative_velocity)
            frame.add(transformed_data)

        return frame

    def to_frame(self, velocity: float) -> "InertialFrame":
        """
        Transforms the frame of reference from being relative to the
        old observer frame to being relative to the new observer frame,
        which has the specified velocity relative to the old observer
        frame.
        """
        frame = self.new_frame(-velocity)
        for data in self.data:
            # Obviously, the data we measure from the perspective of this frame is not dependent
            # on the observer frame. Only when we transform it to the observer frame does the data
            # need to be transformed.
            frame.add(data)
        return frame

    def new_frame(self, relative_velocity: float) -> "InertialFrame":
        """
        Creates a new frame with a given velocity relative to this frame.
        Expects relative velocity as a multiple of c, 
        i.e. relative_velocity_in_mps = c * relative_velocity.
        """
        return InertialFrame(
            velocity_transformation(-self.relative_velocity, relative_velocity)
        )


class SpacetimeDiagram:
    def __init__(self):
        self.frames: List[InertialFrame] = [InertialFrame(0)]
        self.fig = plt.figure()

    def observer_frame(self) -> InertialFrame:
        return self.frames[0]

    def add(self, frame_or_data: Union[InertialFrame, np.ndarray]) -> Optional[int]:
        """
        Adds a new frame to the diagram relative to the observer frame and returns the index of the frame.
        or
        Adds data to the observer frame and returns None.
        """
        if isinstance(frame_or_data, InertialFrame):
            self.frames.append(frame_or_data)
            return len(self.frames) - 1
        else:
            self.frames[0].add(frame_or_data)

    def remove(self, idx: int):
        """
        Removes the inertial frame at the given index.
        If the observer frame is removed, the spacetime diagram will 
        transform to the next frame in the list. If there is only one 
        frame, a `ValueError` is raised, as there is no frame to transform 
        to.
        """
        if len(self.frames) == 1:
            raise ValueError("Cannot remove the observer.")
        if idx != 0:
            self.frames.pop(idx)
        else:
            self.transform(1)
            self.frames.pop(1)

    def transform(self, idx: int):
        """
        Transforms the spacetime diagram to the frame of reference 
        associated with the given index.
        """
        # Get the velocity of the new frame of reference.
        vel = self.frames[idx].relative_velocity
        for i, frame in enumerate(self.frames):
            # Transform data in each frame to the frame with relative velocity vel with respect to the current observer frame.
            self.frames[i] = frame.to_frame(vel)

        # Swap to the new frame
        self.frames[0], self.frames[idx] = self.frames[idx], self.frames[0]

    def build(self, legend=False, marker="o-", **kwargs):
        """
        Builds the spacetime diagram.
        """
        plt.figure(self.fig)
        plt.title("Spacetime Diagram")
        plt.xlabel("Position (x)")
        plt.ylabel("Time (t)")

        for i, frame in enumerate(self.frames):
            for j, data in enumerate(frame.to_observer().data):
                plt.plot(
                    data[:, 1],
                    data[:, 0],
                    marker,
                    label=f"Frame {i} v = {frame.relative_velocity}, Plot {j}",
                    **kwargs,
                )

        plt.grid(True, which="both")
        plt.axhline(0, color="k")
        plt.axvline(0, color="k")

        left, right = plt.xlim()
        top, bottom = plt.ylim()

        width = max(abs(left), abs(right))
        height = max(abs(top), abs(bottom))

        plt.xlim(-width, width)
        plt.ylim(-height, height)

        if legend:
            plt.legend()

    def show(self):
        """
        Displays the spacetime diagram.
        """
        plt.show()

    def get_figure(self) -> plt.Figure:
        return self.fig


__all__ = ["InertialFrame", "SpacetimeDiagram"]

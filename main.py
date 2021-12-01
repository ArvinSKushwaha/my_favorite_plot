import numpy as np
from sptdiag import InertialFrame, SpaceTime
from matplotlib import pyplot as plt


def main():
    # Create a spacetime diagram
    sptdiag = SpaceTime()

    # Create an inertial frame to transform to
    for i in np.linspace(-0.9, 0.9, 100):
        frame = InertialFrame(i)
        frame.add(np.stack([np.linspace(-2, 2, 9), np.zeros(9)], axis=-1))
        sptdiag.add(frame)

    sptdiag.build(marker="ro-", linewidth=0.1)

    fig = sptdiag.get_figure()
    plt.figure(fig)
    plt.plot(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100), "lightgray")
    plt.plot(np.linspace(-10, 10, 100), -np.linspace(-10, 10, 100), "lightgray")
    plt.title("Invariant Interval Hyperbola")

    sptdiag.show()


if __name__ == "__main__":
    main()

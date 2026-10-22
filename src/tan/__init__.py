"""Particle-filter terrain-aided navigation (TAN) for a GPS-denied UAV.

State convention used everywhere in this package::

    x = [px, py, bx, by]
        px, py : position, easting / northing in metres (UTM 44N for real terrain)
        bx, by : INS velocity bias in m/s (true velocity = INS velocity - bias)
"""

__version__ = "0.1.0"

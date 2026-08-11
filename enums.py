from enum import IntEnum


class Direction(IntEnum):
    """Numeric direction values written to the CSV."""

    LEFT = 0
    RIGHT = 1


class VehicleType(IntEnum):
    """Vehicle categories written to the CSV."""

    CAR = 0
    TRUCK = 1
    BUS = 2
    BICYCLE = 3


class TimeOfDay(IntEnum):
    """Lighting modes written to the CSV."""

    DAY = 0
    NIGHT = 1

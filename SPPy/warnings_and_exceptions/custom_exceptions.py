class InvalidSOCException(Exception):
    "Raised when SOC is smaller than 0 or greater than 1."
    def __init__(self, electrode_type):
        self.msg = f"{electrode_type} SOC is beyond 0-1"
        super().__init__(self.msg)


class InsufficientInputOperatingConditions(Exception):
    "Raised when time and current arrays are not present in the input argument."
    pass


class InvalidElectrodeType(Exception):
    "Raised when invalid electrode type is inputted"
    pass


class MaxConcReached(Exception):
    "Raised when maximum concentration is reached"
    pass


class PotientialThesholdReached(Exception):
    "Raised with the threshold potential is reached."
    pass
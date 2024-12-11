from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class OCPParamsBase:
    """Input data structure of the OCP."""

    dt: float # Integration step of the OCP
    T: int # Number of time steps in the horizon
    
    X0: npt.NDArray[np.float64] # Initial state
    


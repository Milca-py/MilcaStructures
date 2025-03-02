

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from core.system import SystemMilcaModel

class GraphicOption:
    def __init__(self,
                system: "SystemMilcaModel",
    ) -> None:
        self.system = system


    @property
    def _length_mean(self):
        return np.mean(
            np.array(
                [
                    element.length
                    for element in self.system.element_map.values()
                    # if element.distributed_load.q_i != 0 or element.distributed_load.q_j != 0
                ]
            )
        )

    @property
    def _qi_mean(self):
        return np.mean(
            np.array(
                [
                    abs(element.distributed_load.q_i)
                    for element in self.system.element_map.values()
                    if element.distributed_load.q_i != 0
                ]
            )
        )
    
    @property
    def _qj_mean(self):
        return np.mean(
            np.array(
                [
                    abs(element.distributed_load.q_j)
                    for element in self.system.element_map.values()
                    if element.distributed_load.q_j != 0
                ]
            )
        )
    

    @property
    def ratio_scale_load(self) -> float:
        return 0.15 * self._length_mean * (2 / (self._qi_mean + self._qj_mean))
        
    @property
    def nrof_arrows(self) -> int:
        # dx = np.trunc(self._length_mean / 10)
        # return np.trunc(self._length_mean / dx)
        return 10

    @property
    def support_size(self) -> float:
        return 0.1 * self._length_mean
    
    @property
    def figsize(self) -> tuple:
        return (10, 10)
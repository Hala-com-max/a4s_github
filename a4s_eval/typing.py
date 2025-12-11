from typing import Union

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]
TextInput = str | list[str]
TextOutput = TextInput

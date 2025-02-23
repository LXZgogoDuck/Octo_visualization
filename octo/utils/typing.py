from typing import Any, Mapping, Sequence, Union

import jax

try:
    PRNGKey = jax.random.KeyArray  # For JAX 0.4.1+
except AttributeError:
    PRNGKey = jax.random.PRNGKey  # For older JAX versions

PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike

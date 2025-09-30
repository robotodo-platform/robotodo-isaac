"""
TODO

"""


import os
import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import jaxtyping as jt
import orbax.checkpoint as ocp


class OpenPiCheckpoint:
    # TODO path maybe url
    def __init__(self, path: os.PathLike):
        self._path = path

    # TODO
    def save(self):
        raise NotImplementedError

    def restore(
        self,
        restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
        dtype: jnp.dtype | None = None,
        sharding: jax.sharding.Sharding | None = None,
    ) -> jt.PyTree:
        """
        Restores unstructured params PyTree from a checkpoint.

        This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
        well as pre-trained checkpoints released for openpi.

        Args:
            # TODO rm: params_path: The local path to the checkpoint directory.
            restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
            dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
            sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

        Returns:
            The restored params.
        """

        params_path = pathlib.Path(self._path).expanduser().resolve()

        if restore_type is jax.Array and sharding is None:
            mesh = jax.sharding.Mesh(jax.devices(), ("x",))
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        with ocp.PyTreeCheckpointer() as checkpoint:
            checkpoint: ocp.PyTreeCheckpointer

            metadata = checkpoint.metadata(params_path)
            item = {"params": metadata["params"]}

            params = checkpoint.restore(
                params_path,
                ocp.args.PyTreeRestore(
                    item=item,
                    restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(
                            sharding=sharding, 
                            restore_type=restore_type, 
                            dtype=dtype,
                        ), 
                        item,
                    ),
                ),
            )["params"]

        # TODO
        # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
        # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
        flat_params = flax.traverse_util.flatten_dict(params)
        if all(kp[-1] == "value" for kp in flat_params):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
        return flax.traverse_util.unflatten_dict(flat_params)

# TODO
class LerobotPi0Checkpoint:
    def restore(self):
        # TODO from safetensors
        raise NotImplementedError


# TODO !!!! necesito??
# from robotodo.controllers.pi0.utils._todo_rm_download import maybe_download

# class Pi0BaseCheckpoint(OpenPiCheckpoint):
#     def __init__(self):
#         # TODO
#         path = maybe_download("gs://openpi-assets/checkpoints/pi0_base")
#         super().__init__(path / "params")

# class Pi0AlohaSimCheckpoint(OpenPiCheckpoint):
#     def __init__(self):
#         # TODO
#         path = maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_sim")
#         super().__init__(path / "params")



import flax.nnx
import jaxtyping as jt
import orbax.checkpoint

import jax._src.tree_util


def check_pytree_equality(*, expected: jt.PyTree, got: jt.PyTree, check_shapes: bool = False, check_dtypes: bool = False):
    """Checks that two PyTrees have the same structure and optionally checks shapes and dtypes. Creates a much nicer
    error message than if `jax.tree.map` is naively used on PyTrees with different structures.
    """

    if errors := list(jax._src.tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + (
                "\n".join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    if check_shapes or check_dtypes:

        def check(kp, x, y):
            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)

def load_from_pytree(
    maybe_abstract_module: flax.nnx.Module,
    *,
    pytree: jt.PyTree, 
    prune: bool = True, 
    strict: bool = True,
):
    graphdef, pytree_src = flax.nnx.split(maybe_abstract_module)
    if prune: # TODO NOTE remove extra unused entries in params
        pytree = orbax.checkpoint.transform_utils.intersect_trees(pytree_src.to_pure_dict(), pytree)
    check_pytree_equality(expected=pytree_src.to_pure_dict(), got=pytree, check_shapes=True, check_dtypes=False)
    if strict:
        def check(kp: str, x: jt.Array, y: jt.Array):
            if x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")
            if x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")
        jax.tree_util.tree_map_with_path(check, pytree_src.to_pure_dict(), pytree)

    # TODO rm: state.replace_by_pure_dict(params)
    flax.nnx.replace_by_pure_dict(pytree_src, pytree)
    return flax.nnx.merge(graphdef, pytree_src)


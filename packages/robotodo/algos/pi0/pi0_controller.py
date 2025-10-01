
import warnings
import dataclasses
import itertools
from types import EllipsisType
from typing import Any, Callable, Protocol, TypedDict, Unpack, Mapping, Hashable, Annotated, Protocol, Literal

import jax
import jaxtyping as jt
import flax.nnx
# TODO
from tensorspecs import TensorTableSpec, rescale
from tensorspecs import TensorSpec, BoxSpec, TensorTableSpec, ShapeLike

from .nn.pi0 import Pi0, Pi0Trainer
from .nn.paligemma_tokenizer import PaligemmaTokenizer


def image_resize_transform(
    batch_image: jt.Float[jt.Array, "batch height width channels"], 
    image_shape: tuple[int, int, int],
    keep_ratio: bool = True,
):
    """
    TODO doc
    TODO test

    .. doctest::

        >>> import jax
        >>> image_resize_transform(
        ...     batch_image=jax.numpy.ones((12, 64, 64, 3)), 
        ...     image_shape=(32, 128, 3),
        ...     keep_ratio=True,
        ... ).shape
        (12, 32, 128, 3)
        >>> image_resize_transform(
        ...     batch_image=jax.numpy.ones((12, 64, 64, 3)), 
        ...     image_shape=(32, 128, 3),
        ...     keep_ratio=False,
        ... ).shape
        (12, 32, 128, 3)

    """

    batch_image_resized = jax.image.resize(
        batch_image, 
        shape=(
            batch_image.shape[0], 
            *(
                (image_shape[0], image_shape[1])
                if not keep_ratio else 
                jax.numpy.array(
                    jax.numpy.array(batch_image.shape)[jax.numpy.array([1, 2])] 
                    * jax.numpy.min(
                        jax.numpy.array(image_shape)[jax.numpy.array([0, 1])]
                        / jax.numpy.array(batch_image.shape)[jax.numpy.array([1, 2])]
                    ),
                    dtype=int,
                )
            ),
            image_shape[2],
        ),
        method="linear",  # TODO make configurable
    )

    batch_image_resized = jax.numpy.pad(
        batch_image_resized,
        pad_width=(
            (0, 0), 
            (0, image_shape[0] - batch_image_resized.shape[1]), 
            (0, image_shape[1] - batch_image_resized.shape[2]), 
            (0, 0),
        ),
        mode="constant",
        constant_values=0.0,
    )

    # TODO
    if image_shape[2] != batch_image_resized.shape[3]:
        raise NotImplementedError
    
    return batch_image_resized


# TODO
class Pi0Controller:
    """
    TODO doc

    """

    class ObservationSpec(TensorTableSpec):
        """
        Observation specification.

        TODO doc
        
        """

        def __init__(
            self,
            specs: Mapping[
                Literal["text"] 
                | tuple[Literal["camera", "joint"], Hashable], 
                TensorSpec,
            ] = {
                "text": TensorSpec(dtype=str),
                **{
                    ("camera", key): BoxSpec(
                        # TODO any order
                        "height width channels",
                        bounds=(0., 1.),
                    )
                    for key in [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb",
                    ]
                },
                **{
                    ("joint", key): TensorSpec()
                    for key in range(32)
                },
            },
        ):
            # TODO validate
            super().__init__(
                TensorTableSpec(specs, "batch?")
                .expand_dims("batch?")
            )
            # super().__init__(specs, dims="batch?")

    class ObservationSpecTransform:
        class _ImageTransformCallable(Protocol):
            def __call__(
                self, 
                batch_image: jt.Float[jt.Array, "batch height width channels"], 
                target_shape: tuple[int, int, int],
            ) -> jt.Float[jt.Array, "batch height width channels"]:
                ...

        class _TextTokenizerCallable(Protocol):
            def __call__(
                self, 
                batch_text: list[str], 
                max_len: int,
            ) -> list[jt.Int[jt.Array, "token"]]:
                ...

        @staticmethod
        def encode(
            # TODO
            target_spec: "Pi0.ObservationSpec",
            source_spec: "Pi0Controller.ObservationSpec",
            source: ...,
            image_transform: _ImageTransformCallable,
            text_tokenizer: _TextTokenizerCallable,
        ):
            """
            TODO doc
            """
            
            source = source_spec.reshape({"batch?": -1}, source)

            num_tokens_max = 0
            batch_tokens = text_tokenizer(source["text"])
            for batch_i, tokens in enumerate(batch_tokens):
                num_tokens_max = max(num_tokens_max, len(tokens))

            num_cameras_max = 0
            for key in source_spec:
                match key:
                    case "camera", _:
                        num_cameras_max += 1

            # preallocate+prefill to make indexing easier
            target = target_spec.reshape({
                # TODO check if already set; DO NOT OVERRIDE UNLESS None!!!!!
                "batch": source_spec.shape_of(source)["batch?"],
                "token": num_tokens_max,
                "image": num_cameras_max,
            }).empty()
            for key, fill_value in [
                ("images.mask", False),
                ("tokenized_text.mask", False),
                ("numeric_state", 0.),
            ]:
                target[key] = target_spec[key].set(target[key], fill_value)

            camera_index = 0
            joint_index = 0
            
            for key in source_spec:
                match key:
                    case "text":
                        if key not in source:
                            continue

                        for batch_i, tokens in enumerate(batch_tokens):
                            if ((source_len := len(tokens)) 
                                > (target_len := len(target["tokenized_text"][batch_i]))):
                                warnings.warn(
                                    f"Tokenized text at batch index {batch_i} "
                                    f"has length {source_len} > {target_len}, truncating: "
                                    f"{tokens}"
                                )
                                tokens = tokens[:target_len]
                            for k, fill_value in [
                                ("tokenized_text", tokens),
                                ("tokenized_text.mask", True),
                            ]:
                                target[k] = target_spec[k].set(
                                    target[k],
                                    fill_value,
                                    indices={"batch": batch_i, "token": slice(0, len(tokens))},
                                )

                    case "camera", _:
                        if key not in source:
                            camera_index += 1
                            continue

                        image = source[key]
                        image = image_transform(
                            image,
                            # TODO
                            tuple(
                                target_spec["images"].shape[dim]
                                for dim in ("height", "width", "channels")
                            ),
                        )

                        if any(
                            not isinstance(d, BoxSpec) 
                            for d in (target_spec["images"], source_spec[key])
                        ):
                            warnings.warn(
                                f"Image rescaling skipped for camera '{key}' due to non-BoxSpec: "
                                f"""{target_spec["images"]}, {source_spec[key]}"""
                            )
                        else:
                            image = rescale(
                                target_spec=target_spec["images"],
                                source_spec=source_spec[key],
                                source=image,
                            )

                        for k, fill_value in [
                            ("images", image),
                            ("images.mask", True),
                        ]:
                            target[k] = target_spec[k].set(
                                target[k],
                                fill_value,
                                indices={"image": camera_index},
                            )
                        camera_index += 1

                    case "joint", _:
                        if key not in source:
                            # TODO raise?
                            joint_index += 1
                            continue

                        # TODO check joint_index overflow

                        target["numeric_state"] = target_spec["numeric_state"].set(
                            target["numeric_state"],
                            source[key],
                            indices={"state": joint_index},
                        )
                        joint_index += 1

            # TODO
            return target
        
    class ActionSpec(TensorTableSpec):
        """
        Action specification.

        TODO doc
        
        """

        def __init__(
            self,
            specs: Mapping[
                tuple[Literal["joint"], Hashable], 
                TensorSpec,
            ] = {
                ("joint", key): TensorSpec()
                for key in range(32)
            }
        ):
            # TODO validate
            super().__init__(
                TensorTableSpec(specs, "batch? timestep")
                .expand_dims("batch? timestep")
            )
            # super().__init__(specs, dims="batch? timestep")

    class ActionSpecTransform:
        @staticmethod
        def encode(
            # TODO
            target_spec: "Pi0.ActionSpec",
            source_spec: "Pi0Controller.ActionSpec",
            source: ...,
        ):
            """
            TODO doc
            """

            source = source_spec.reshape({"batch?": -1}, source)

            # preallocate+prefill to make indexing easier
            target = target_spec.reshape({
                "batch": source_spec.shape_of(source)["batch?"],
                "timestep": source_spec.shape_of(source)["timestep"],
            }).empty()
            for key, fill_value in [
                ("numeric_states", 0.)
            ]:
                target[key] = target_spec[key].set(target[key], fill_value)

            joint_index = 0

            for key in source_spec:
                match key:
                    case "joint", _:
                        if key not in source:
                            # TODO raise?
                            joint_index += 1
                            continue

                        # TODO check joint_index overflow

                        target["numeric_states"] = target_spec["numeric_states"].set(
                            target["numeric_states"],
                            source[key],
                            indices={"state": joint_index},
                        )
                        joint_index += 1

            return target
        
        @staticmethod
        def decode(
            # TODO
            target_spec: "Pi0Controller.ActionSpec",
            source_spec: "Pi0.ActionSpec",
            source: ...,
        ):
            """
            TODO doc
            """

            target = target_spec.reshape({
                "batch?": source_spec.shape_of(source)["batch"],
                "timestep": source_spec.shape_of(source)["timestep"],
            }).empty()

            joint_index = 0

            for key in target_spec:
                match key:
                    case "joint", _:
                        target[key] = target_spec[key].set(
                            target[key],
                            source_spec["numeric_states"].get(
                                source["numeric_states"], 
                                indices={"state": joint_index}
                            )
                        )
                        joint_index += 1

            return target

    class Config(TypedDict):
        observation_spec: "Pi0Controller.ObservationSpec"
        action_spec: "Pi0Controller.ActionSpec"
        # TODO
        nn: Pi0 | None
        nn_trainer: Pi0Trainer | None

    # TODO 
    def __init__(
        self, 
        config: Config = Config(
            observation_spec=ObservationSpec(),
            action_spec=ActionSpec(),
        ), 
        **kwds: Unpack[Config],
    ):
        """
        TODO doc
        """

        config = self.Config(config, **kwds)

        # TODO validate
        self._action_spec = config["action_spec"]
        self._observation_spec = config["observation_spec"]

        self._text_tokenizer = PaligemmaTokenizer()

        self._nn = config.get("nn", None)
        if self._nn is None:
            # TODO custom config !!!!!
            self._nn = Pi0(
                rngs=flax.nnx.Rngs(0),  # TODO
                # config=Pi0.Config(...),
            )

        self._nn_trainer = config.get("nn_trainer", None)
        if self._nn_trainer is None:
            self._nn_trainer = Pi0Trainer(self._nn)

    # TODO
    @classmethod
    def from_checkpoint(cls, resource: str):

        config = ...
        match resource:
            case "pi0_base":
                raise NotImplementedError("TODO WIP")
                # TODO
                # config = cls.Config(
                #     nn=Pi0.from_checkpoint("pi0_base"),
                # )
            case "pi0_aloha_sim":
                camera_keys = [
                    ("camera", name)
                    for name in ["head", "wrist.l", "wrist.r"]
                ]
                joint_keys = [
                    *(("joint", ("arm.l", index)) for index in range(6)),
                    ("joint", "gripper.l"),
                    *(("joint", ("arm.r", index)) for index in range(6)),
                    ("joint", "gripper.r"),
                ]
                
                observation_spec = cls.ObservationSpec({
                    "text": TensorSpec(dtype=str),
                    **{
                        camera_key: BoxSpec(
                            "height width channels",
                            bounds=(0., 1.),
                        )
                        for camera_key in camera_keys
                    },
                    **{
                        joint_key: TensorSpec()
                        for joint_key in joint_keys
                    },
                })
                action_spec = cls.ActionSpec({
                    joint_key: TensorSpec()
                    for joint_key in joint_keys
                })

                config = cls.Config(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    nn=Pi0.from_checkpoint("pi0_aloha_sim"),
                )
                return cls(config)
                
            case _:
                raise NotImplementedError(f"TODO FIXME WIP Currently not supported: {resource}")
        
        # TODO
        raise NotImplementedError
    
    @property
    def observation_spec(self):
        return self._observation_spec
    
    @property
    def action_spec(self):
        return self._action_spec

    @property
    def nn(self):
        return self._nn
    
    # TODO FIXME perf: jit the whole thing
    # TODO Impl[ObservationSpec]
    def compute_action(
        self, 
        maybe_batch_observation: ...,
        rng: jt.PRNGKeyArray | None = None,
        num_timesteps: int = 1,
        diffusion_num_steps: int = 10,
    ):
        """
        Compute action from observation.

        :param maybe_batch_observation: Observation, possibly batched.
        :param rng: JAX random number generator created by :func:`jax.random.key`.
        :param num_timesteps: Number of timesteps to predict.
        :param diffusion_num_steps: Number of diffusion steps.
        :return: Action, batched if observation is batched.

        Example:

        .. doctest::

            >>> # TODO
        

        """

        # auto-batching
        observation_shape = (
            self._observation_spec
            .shape_of(maybe_batch_observation)
        )
        batch_observation = self._observation_spec.reshape(
            {"batch?": -1},
            maybe_batch_observation,
        )

        batch_observation_nn = Pi0Controller.ObservationSpecTransform.encode(
            target_spec=self._nn.observation_spec,
            source_spec=self._observation_spec,
            source=batch_observation,
            image_transform=image_resize_transform,
            text_tokenizer=self._text_tokenizer,
        )

        if rng is None:
            rng = jax.random.key(0)

        batch_actions_nn = self._nn.compile_jit(
            self._nn.sample_actions,
            static_argnames=("num_timesteps", "denoising_num_steps"),
        )(
            rng=rng, 
            batch_observation=batch_observation_nn,
            num_timesteps=num_timesteps,
            denoising_num_steps=diffusion_num_steps,
        )

        batch_action = Pi0Controller.ActionSpecTransform.decode(
            target_spec=self._action_spec,
            source_spec=self._nn.action_spec,
            source=batch_actions_nn,
        )

        # auto-batching
        return self._action_spec.reshape(
            {"batch?": observation_shape["batch?"]},
            batch_action,
        )

    # TODO !!!!!
    def learn(
        self, 
        maybe_batch_observation: ...,
        maybe_batch_action: ...,
        rng: jt.PRNGKeyArray | None = None,
        training_num_steps: int = 1,
    ):
        """
        Learn from observation-action pairs.
        
        :param maybe_batch_observation: Observation, possibly batched.
        :param maybe_batch_action: Action, possibly batched.
        :param rng: JAX random number generator created by :func:`jax.random.key`.
        :param training_num_steps: Number of training steps.
        
        Example:

        .. doctest::
            >>> # TODO

        """
        
        batch_observation = self._observation_spec.reshape(
            {"batch?": -1},
            maybe_batch_observation,
        )
        batch_action = self._action_spec.reshape(
            {"batch?": -1},
            maybe_batch_action,
        )

        # TODO assert same batch size!

        batch_observation_nn = Pi0Controller.ObservationSpecTransform.encode(
            target_spec=self._nn.observation_spec,
            source_spec=self._observation_spec,
            source=batch_observation,
            image_transform=image_resize_transform,
            text_tokenizer=self._text_tokenizer,
        )
        batch_action_nn = Pi0Controller.ActionSpecTransform.encode(
            source_spec=self._action_spec,
            source=batch_action,
            target_spec=self._nn.action_spec,
        )

        if rng is None:
            rng = jax.random.key(0)

        # TODO training stats
        for i in range(training_num_steps):
            rng_step = jax.random.fold_in(rng, i)
            self._nn_trainer.step(
                rng=rng_step,
                batch_observation=batch_observation_nn,
                batch_action=batch_action_nn,
            )

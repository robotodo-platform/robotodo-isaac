"""
TODO

"""

# TODO ref pi0!!!!


import functools
import re
# TODO
import dataclasses
from typing import Any, Mapping, Optional, TypeVar, Callable, ParamSpec # TODO !!!!!


# TODO rm
# import gymnasium.spaces
# import numpy

from tensorspecs import TensorSpec, BoxSpec, TensorTableSpec, TensorTableLike, ShapeLike

_P = ParamSpec("_P")
_R = TypeVar("_R")

# ######################################


import optax
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import jaxtyping as jt

from . import gemma as _gemma
from . import siglip as _siglip

# TODO
from .utils import PathRegexFilter, nnx_frozen_jit
    

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


# @jt.typecheck
def posemb_sincos(
    pos: jt.Real[jt.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> jt.Float[jt.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(nnx.Module):
    """
    TODO

    Example:
    .. code-block:: python

        # TODO
    
    """

    class ObservationSpec(TensorTableSpec):
        def __init__(
            self,
            image_shape: ShapeLike = {
                "height": 224,
                "width": 224,
                "channels": 3,
            },
            numeric_state_size: int = 32,
            image_state_size: int | None = None,
            text_state_size: int | None = None,
        ):
            super().__init__({
                "images": BoxSpec(
                    "batch image height width channels",
                    shape={**image_shape, "image": image_state_size},
                    dtype="float32",
                    bounds=(-1, 1),
                ),
                "images.mask": TensorSpec(
                    "batch image",
                    shape={"image": image_state_size},
                    dtype="bool",
                ),
                "tokenized_text": TensorSpec(
                    "batch token",
                    shape={"token": text_state_size},
                    dtype="int32",
                ),
                "tokenized_text.mask": TensorSpec(
                    "batch token",
                    shape={"token": text_state_size},
                    dtype="bool",
                ),
                "numeric_state": TensorSpec(
                    "batch state",
                    shape={"state": numeric_state_size},
                    dtype="float32",
                ),
            }, "batch")

    class ActionSpec(TensorTableSpec):
        def __init__(
            self,
            numeric_state_size: int = 32,
            num_timesteps: int | None = None,
        ):
            super().__init__({
                "numeric_states": TensorSpec(
                    "batch timestep state",
                    shape={
                        "state": numeric_state_size,
                        "timestep": num_timesteps,
                    },
                    dtype="float32",
                ),
            }, "batch timestep")

    # TODO make it TypedDict to support pytree?
    @dataclasses.dataclass(frozen=True)
    class Config:
        # TODO
        image_shape: Mapping[str, int] = dataclasses.field(default_factory=lambda: {
            "height": 224,
            "width": 224,
            "channels": 3,
        })
        """TODO doc"""
        numeric_state_size: int = 32
        """TODO doc"""

        dtype: str = "bfloat16"
        paligemma_variant: _gemma.Variant = "gemma_2b"
        action_expert_variant: _gemma.Variant = "gemma_300m"

    def __init__(
        self, 
        rngs: nnx.Rngs,
        config: Config = Config(), 
    ):
        """
        TODO doc

        """

        self._config = config

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # TODO: rewrite gemma in NNX. For now, use bridge.
        self.gemma_language_model = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        # TODO
        self.gemma_language_model.lazy_init(rngs=rngs, method="init")

        self.siglip_vision_model = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # TODO FIXME: this keeps recompiling!!!!!
        self.siglip_vision_model.lazy_init(
            # TODO NOTE the batch size isnt actually correct; but the model only cares about the image shape
            jnp.empty((1, *(config.image_shape[key] for key in ["height", "width", "channels"]))),
            train=False,
            rngs=rngs,
            method="init",
        )

        self.state_proj = nnx.Linear(config.numeric_state_size, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.numeric_state_size, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.numeric_state_size, rngs=rngs)

        # NOTE compat
        self.PaliGemma = nnx.Dict(
            llm=self.gemma_language_model, 
            img=self.siglip_vision_model,
        )

    # TODO 
    @classmethod
    def from_checkpoint(cls, resource: str) -> "Pi0":
        """
        TODO doc

        currently only literals are supported!!!!
        also we need to allow storing configs in the ckpt!!!!
        """

        # TODO FIXME these absolutely DO NOT BELONG HERE!!!
        from ..utils._todo_rm_download import maybe_download
        from ..utils._todo_checkpoint import load_from_pytree, OpenPiCheckpoint        

        pi0_pytree = ...
        match resource:
            case "pi0_base":
                pi0_pytree = OpenPiCheckpoint(
                    maybe_download("gs://openpi-assets/checkpoints/pi0_base")
                    / "params"
                ).restore()
            case "pi0_aloha_sim":
                pi0_pytree = OpenPiCheckpoint(
                    maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_sim")
                    / "params"
                ).restore()
            case _:
                # TODO
                raise NotImplementedError(f"TODO FIXME WIP Currently not supported!!: {resource}")

        abstract_pi0 = nnx.eval_shape(lambda cls=cls: cls(rngs=nnx.Rngs(0)))
        return load_from_pytree(abstract_pi0, pytree=pi0_pytree)

    # TODO
    def checkpoint(self, resource):
        raise NotImplementedError

    # TODO necesito?
    # @property
    # def config(self) -> Config:
    #     return self._config

    @property
    def observation_spec(self):
        """
        TODO doc

        .. seealso::
            :class:`Pi0.ObservationSpec`
        """

        return self.ObservationSpec(
            image_shape=self._config.image_shape,
            numeric_state_size=self._config.numeric_state_size,
        )

    @property
    def action_spec(self):
        """
        TODO doc

        .. seealso::
            :class:`Pi0.ActionSpec`
        """

        return self.ActionSpec(
            numeric_state_size=self._config.numeric_state_size,
        )

    # @jt.typecheck
    # @nnx.jit
    def _embed_prefix(
        self, 
        # TODO typing!!!!
        batch_observation: TensorTableLike[ObservationSpec],
    ) -> tuple[jt.Float[jt.Array, "b s emb"], jt.Bool[jt.Array, "b s"], jt.Bool[jt.Array, " s"]]:
        # TODO doc: batch(variable) tokens_len(variable) embedding_size
        # TODO 

        embeddings = []
        input_masks = []
        ar_masks = []

        # TODO NOTE !!!!!! batch is the first dim !!!!!
        # embed images
        for image_index in range(batch_observation["images"].shape[1]):
            # TODO FIXME optimize: embed all images at once!!!
            image_embeddings, _ = self.siglip_vision_model(
                # TODO
                batch_observation["images"][:, image_index], 
                train=False,
            )

            embeddings.append(image_embeddings)

            # TODO ...
            # einops.parse_shape(image_embeddings.shape)

            input_masks.append(
                jnp.full(
                    (image_embeddings.shape[0], image_embeddings.shape[1]), 
                    True, 
                    dtype=jnp.bool,
                )
                if batch_observation.get("images.mask", None) is None else
                einops.repeat(
                    batch_observation["images.mask"][:, image_index],
                    "b -> b s",
                    b=image_embeddings.shape[0],
                    s=image_embeddings.shape[1],
                )
            )

            # image tokens attend to each other
            # TODO
            ar_masks.append(jnp.full(image_embeddings.shape[1], False))

        # add language (aka tokenized inputs)
        if batch_observation.get("tokenized_text", None) is not None:
            language_embeddings = self.gemma_language_model(
                # TODO
                batch_observation["tokenized_text"], 
                method="embed",
            )

            embeddings.append(language_embeddings)

            input_masks.append(
                jnp.full(batch_observation["tokenized_text"], True, dtype=jnp.bool)
                if batch_observation.get("tokenized_text.mask", None) is None else
                batch_observation["tokenized_text.mask"]
            )

            # full attention between image and language inputs
            ar_masks.append(jnp.full(language_embeddings.shape[1], False))

        embeddings = jnp.concatenate(embeddings, axis=1)
        input_masks = jnp.concatenate(input_masks, axis=1)
        ar_masks = jnp.concatenate(ar_masks)

        return embeddings, input_masks, ar_masks

    # @jt.typecheck
    # TODO typing !!~!!!!!
    # @nnx.jit
    def _embed_suffix(
        self, 
        # TODO typing !!!!
        batch_observation: TensorTableLike[ObservationSpec],
        batch_noisy_action: TensorTableLike[ActionSpec],
        timestep: jt.Float[jt.Array, " b"],
    ) -> tuple[jt.Float[jt.Array, "b s emb"], jt.Bool[jt.Array, "b s"], jt.Bool[jt.Array, " s"]]:
        _, action_horizon, _ = batch_noisy_action["numeric_states"].shape

        input_mask = []
        ar_mask = []
        tokens = []

        # add a single state token
        state_token = self.state_proj(batch_observation["numeric_state"])[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((batch_observation["numeric_state"].shape[0], 1), dtype=jnp.bool_))

        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(batch_noisy_action["numeric_states"])
        # TODO !!!!
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    # TODO
    # @override
    # @nnx_frozen_jit
    def sample_actions(
        self,
        rng: jt.PRNGKeyArray,
        batch_observation: TensorTableLike[ObservationSpec],
        num_timesteps: int = 1,
        denoising_num_steps: int | jt.Int[jt.Array, ""] = 10,
    ) -> TensorTableLike[ActionSpec]:
        """
        Compute actions given a batch of observations.

        :param rng: JAX PRNG key for random number generation.
        :param batch_observation: Batch of observations to compute actions for.
        :param num_timesteps: Number of action timesteps.
        :param denoising_num_steps: Number of denoising steps.
        :return: Batch of actions.

        Example:
        .. testcode ::

            import jax
            import flax.nnx

            pi0 = Pi0(flax.nnx.Rngs(0))
            pi0.sample_actions(
                jax.random.key(0),
                pi0.observation_spec.random(),
            )
        
        """

        # TODO rm
        # observation = _model.preprocess_observation(None, observation, train=False)
        # ###############

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / denoising_num_steps
        batch_size, *_ = batch_observation["numeric_state"].shape
        # TODO rm?
        # action_horizon, action_dim = self._config.action_spec["numeric_states"].shape

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix(batch_observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.gemma_language_model([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry: tuple[float, float]):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self._embed_suffix(
                batch_observation, {"numeric_states": x_t}, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.gemma_language_model(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -num_timesteps:])

            return x_t + dt * v_t, time + dt

        def cond(carry: tuple[float, float]):
            _, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        noise = jax.random.normal(rng, (batch_size, num_timesteps, self.action_out_proj.out_features))
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return {"numeric_states": x_0}
    
    # TODO typing
    # @override
    # @nnx_frozen_jit
    def compute_loss(
        self, 
        rng: jt.PRNGKeyArray,
        batch_observation: TensorTableLike[ObservationSpec], 
        batch_action: TensorTableLike[ActionSpec], 
        # train: bool = False,
    ) -> jt.Float[jt.Array, "*b ah"]:
        # TODO
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # TODO rm
        # observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_size, action_horizon, _ = batch_action["numeric_states"].shape

        noise = jax.random.normal(noise_rng, batch_action["numeric_states"].shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size, )) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * batch_action["numeric_states"]
        u_t = noise - batch_action["numeric_states"]

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix(batch_observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self._embed_suffix(batch_observation, {"numeric_states": x_t}, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.gemma_language_model(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.action_out_proj(suffix_out[:, -action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    @functools.cache
    def compile_jit(self, method: Callable[_P, _R], **jax_jit_kwds) -> Callable[_P, _R]:
        return nnx_frozen_jit(method, **jax_jit_kwds)
    
    # TODO
    @functools.cached_property
    def trainable_filter(self):
        config: Pi0.Config = self._config

        filters = []
        has_lora = False
        gemma_params_filter = PathRegexFilter(".*llm.*")
        action_expert_params_filter = PathRegexFilter(".*llm.*_1.*")

        if "lora" in config.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in config.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in config.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(PathRegexFilter(".*lora.*")),
            )

        if not filters:
            return nnx.Nothing
        
        return nnx.All(nnx.Param, nnx.Not(nnx.All(*filters)))


class Pi0LearningRateScheduleRecipes:
    @classmethod
    def warmup_cosine_decay_schedule(
        cls,
        warmup_steps: int = 1_000,
        peak_lr: float = 2.5e-5,
        decay_steps: int = 30_000,
        decay_lr: float = 2.5e-6,
    ):
        return optax.warmup_cosine_decay_schedule(
            init_value=peak_lr / (warmup_steps + 1),
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=decay_lr,
        )
    
    @classmethod
    def rsqrt_decay_schedule(
        cls,
        warmup_steps: int = 1_000,
        peak_lr: float = 5e-5,
        timescale: float = 10_000,
    ):
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=peak_lr / (warmup_steps + 1),
                    end_value=peak_lr,
                    transition_steps=warmup_steps,
                ),
                lambda step: peak_lr / jnp.sqrt((timescale + step) / timescale),
            ],
            [warmup_steps],
        )

class Pi0OptimizerRecipes:
    @classmethod
    def adamw(
        cls, 
        clip_gradient_norm: float = 1., 
        **optax_adamw_kwargs,
    ):
        return optax.chain(
            optax.clip_by_global_norm(clip_gradient_norm), 
            optax.adamw(**{
                "b1": 0.9,
                "b2": 0.95,
                "eps": 1e-8,
                "weight_decay": 1e-10,
                **optax_adamw_kwargs,
            })
        )

    @classmethod
    def sgd(cls, **optax_sgd_kwargs):
        return optax.sgd(**{
            "learning_rate": 5e-5,
            "momentum": 0.9,
            "nesterov": False,
            **optax_sgd_kwargs,
        })


import functools

# TODO rename to Pi0SupervisedTrainer? Pi0ReinforcementTrainer?
class Pi0Trainer(nnx.Module):
    """
    TODO
    
    Example usage:
    .. testcode:: python
        # TODO

    .. code-block:: python
    
        import jax
        import optax

        model = Pi0()
        trainer = Pi0Trainer(
            config=Pi0Trainer.Config(
                optimizer=optax.adam(1e-3),
            )
        )

        rng = jax.random.key(0)
        for _ in range(1):
            trainer.step(
                rng=rng,
                # TODO !!!!!
                batch_observation=pi0.observation_spec().sample(),
                batch_action=pi0.action_spec().sample(),
            )

    """

    @dataclasses.dataclass(frozen=True)
    class Config:
        optimizer: optax.GradientTransformation = Pi0OptimizerRecipes.sgd()

    @dataclasses.dataclass
    class State:
        optimizer_state: optax.OptState
        # step_n: int = 0

    def __init__(self, model: Pi0, config: Config = Config()):
        self._model = model
        self._optimizer = config.optimizer
        # TODO lazy
        # self._state = self.State(
        #     optimizer_state=self._optimizer.init(
        #         nnx.state(self._model, self._model.trainable_filter),
        #     ),
        #     # step_n=0,
        # )

    @functools.cached_property
    def _state(self):
        return self.State(
            optimizer_state=self._optimizer.init(
                nnx.state(self._model, self._model.trainable_filter),
            ),
            # step_n=0,
        )
    
    # TODO
    @classmethod
    def from_checkpoint(cls, resource) -> "Pi0Trainer":
        raise NotImplementedError

    # TODO
    def checkpoint(self, resource):
        raise NotImplementedError

    @dataclasses.dataclass(frozen=True)
    class StepResult:
        loss: jt.Array
        grads: jt.Array

    def step(
        self,
        rng: jt.PRNGKeyArray,
        batch_observation: TensorTableLike[Pi0.ObservationSpec],
        batch_action: TensorTableLike[Pi0.ActionSpec],
    ) -> StepResult:

        # TODO
        # rng_train_step = jax.random.fold_in(rng, self._state.step_n)

        # @at.typecheck
        def loss_fn(
            model: Pi0, 
            rng: jt.PRNGKeyArray, 
            batch_observation: TensorTableLike[Pi0.ObservationSpec], 
            batch_action: TensorTableLike[Pi0.ActionSpec],
        ):
            chunked_loss = model.compute_loss(
                batch_observation=batch_observation, 
                batch_action=batch_action,
                rng=rng, 
            )
            return jnp.mean(chunked_loss)
        # TODO
        self._model.train(mode=True)
        loss, grads = nnx.value_and_grad(
            loss_fn,
            # TODO simplify?
            argnums=nnx.DiffState(0, self._model.trainable_filter),
        )(
            self._model, 
            rng=rng,
            # rng=rng_train_step, 
            batch_observation=batch_observation, 
            batch_action=batch_action,
        )

        trainable_params = nnx.state(self._model, self._model.trainable_filter)
        optimizer_updates, self._state.optimizer_state = self._optimizer.update(
            grads,
            self._state.optimizer_state,
            params=trainable_params,
        )
        trainable_params = optax.apply_updates(trainable_params, optimizer_updates)
        nnx.update(self._model, trainable_params)

        # self._state.step_n += 1

        return Pi0Trainer.StepResult(
            loss=loss,
            grads=grads,
        )

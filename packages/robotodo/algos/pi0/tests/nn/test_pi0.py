"""
TODO doc

"""


import pytest
import jax
import flax.nnx
import openpi.models.pi0
import openpi.models.pi0_config
import openpi.models.model

from robotodo.algos.pi0.nn.pi0 import Pi0


@pytest.mark.parametrize(
    "openpi_config", 
    [
        openpi.models.pi0_config.Pi0Config(
            action_dim=4, 
            action_horizon=2, 
            max_token_len=8, 
            dtype="bfloat16", 
            paligemma_variant="dummy", 
            action_expert_variant="dummy",
            pi05=False,
        ),
        openpi.models.pi0_config.Pi0Config(
            action_dim=4, 
            action_horizon=2, 
            max_token_len=8, 
            dtype="bfloat16", 
            paligemma_variant="dummy", 
            action_expert_variant="dummy",
            pi05=True,
        ),
    ],
)
@pytest.mark.parametrize(
    "rng", 
    [
        jax.random.key(0),
    ],
)
@pytest.mark.parametrize(
    "should_copy_weights", 
    [
        True,
        False,
    ],
)
def test_pi0_openpi_compliance(openpi_config, rng, should_copy_weights):

    def make_init_rngs():
        # NOTE due to determinism this cannot be reused
        # `flax.nnx.Rngs` is not pure because every call mutates its internal states
        return flax.nnx.Rngs(rng)

    def make_rng():
        return rng
    
    openpi0_config: openpi.models.pi0_config.Pi0Config = openpi_config

    openpi0 = openpi.models.pi0.Pi0(
        openpi0_config, 
        rngs=make_init_rngs(),
    )

    observation_openpi0 = openpi0_config.fake_obs()
    # NOTE this prevents preprocessing built into the openpi model 
    # from modifying the input (e.g. normalization)
    observation_openpi0 = openpi.models.model.preprocess_observation(
        None, observation_openpi0, train=False,
    )
    action_openpi0 = openpi0_config.fake_act()

    action_horizon = action_openpi0.shape[-2]
    denoising_num_steps = 10

    pi0 = Pi0(
        rngs=make_init_rngs(),
        config=Pi0.Config(
            image_shape={
                "height": observation_openpi0.images["base_0_rgb"].shape[-3], 
                "width": observation_openpi0.images["base_0_rgb"].shape[-2], 
                "channels": observation_openpi0.images["base_0_rgb"].shape[-1], 
            },
            numeric_state_size=observation_openpi0.state.shape[-1],
            dtype=openpi0_config.dtype,
            variant="pi05" if openpi0.pi05 else "pi0",
            paligemma_variant=openpi0_config.paligemma_variant,
            action_expert_variant=openpi0_config.action_expert_variant,
        ),
    )
    # TODO
    if should_copy_weights:
        pi0: Pi0 = flax.nnx.merge(flax.nnx.graphdef(pi0), flax.nnx.state(openpi0))

    observation = pi0.observation_spec.reshape(sizes={"image": len(observation_openpi0.images)}).empty()
    observation["images"] = jax.numpy.asarray(observation["images"])
    observation["images.mask"] = jax.numpy.asarray(observation["images.mask"])
    for i, (image_key, image) in enumerate(observation_openpi0.images.items()):
        observation["images"] = observation["images"].at[:, i].set(image, mode="fill")
        observation["images.mask"] = observation["images.mask"].at[:, i].set(
            observation_openpi0.image_masks[image_key],
            mode="fill",
        )
    observation["tokenized_text"] = observation_openpi0.tokenized_prompt
    observation["tokenized_text.mask"] = observation_openpi0.tokenized_prompt_mask
    observation["numeric_state"] = observation_openpi0.state

    action = pi0.action_spec.empty()
    action["numeric_states"] = action_openpi0

    # .embed_prefix
    for openpi0_embeddings, pi0_embeddedings in zip(
        openpi0.embed_prefix(observation_openpi0), 
        pi0.embed_prefix(observation),
    ):
        assert jax.numpy.array_equiv(openpi0_embeddings, pi0_embeddedings)

    # .embed_suffix
    batch_size = observation_openpi0.state.shape[0]
    timestep = jax.numpy.broadcast_to(0., batch_size)
    for name, openpi0_embeddings, pi0_embeddedings in zip(
        ["tokens", "input_mask", "ar_mask", "adarms_cond"],
        openpi0.embed_suffix(observation_openpi0, action_openpi0, timestep=timestep), 
        pi0.embed_suffix(observation, action, timestep=timestep),
    ):
        if openpi0_embeddings is None or pi0_embeddedings is None:
            assert openpi0_embeddings == pi0_embeddedings
        else:
            assert jax.numpy.array_equiv(openpi0_embeddings, pi0_embeddedings), name

    # .compute_loss
    computed_loss_openpi0 = openpi0.compute_loss(
        rng=make_rng(),
        observation=observation_openpi0,
        actions=action_openpi0,
    )
    computed_loss = pi0.compute_loss(
        rng=make_rng(),
        batch_observation=observation,
        batch_action=action,
    )
    assert jax.numpy.array_equiv(computed_loss_openpi0, computed_loss)

    # .sample_actions
    computed_actions_openpi0 = openpi0.sample_actions(
        rng=make_rng(),
        observation=observation_openpi0,
        num_steps=denoising_num_steps,
    )
    computed_action = pi0.sample_actions(
        rng=make_rng(),
        batch_observation=observation,
        num_timesteps=action_horizon,
        denoising_num_steps=denoising_num_steps,
    )
    assert jax.numpy.array_equiv(computed_actions_openpi0, computed_action["numeric_states"])


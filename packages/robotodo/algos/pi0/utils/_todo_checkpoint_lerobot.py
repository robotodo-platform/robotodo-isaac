

# TODO
from typing import Mapping

import einops
import jax.numpy
import jaxtyping as jt

from ..nn.pi0 import Pi0


def lerobot_transform_state_dict(pi0_le: dict) -> dict:
    """
    Lerobot sucks.
    Lerobot sucks.
    Lerobot sucks.
    Lerobot SUCKS!!!!

    Transform state dict keys to match expected model structure.
    FOR COMPAT PURPOSES ONLY.
    
    Transformations:
    - model.paligemma_with_expert.paligemma.language_model.lm_head ->
        model.paligemma_with_expert.paligemma.lm_head
    - model.paligemma_with_expert.paligemma.language_model.model ->
        model.paligemma_with_expert.paligemma.model.language_model
    - model.paligemma_with_expert.paligemma.vision_tower ->
        model.paligemma_with_expert.paligemma.model.vision_tower
    - model.paligemma_with_expert.paligemma.multi_modal_projector ->
        model.paligemma_with_expert.paligemma.model.multi_modal_projector

    Also handles tied weights between lm_head.weight and
    embed_tokens.weight.

    :param pi0_le: The Lerobot state dict to transform.
    :return: A copy of the state dict with keys matching expected model structure.

    .. seealso ::
        https://github.com/huggingface/lerobot/blob/55198de096f46a8e0447a8795129dd9ee84c088c/src/lerobot/policies/pi0/modeling_pi0.py#L257C1-L326C32

    """
    import re

    transformed_dict = {}

    transformations = [
        (
            re.compile(r"\.paligemma_with_expert\.paligemma\.language_model\.lm_head"),
            ".paligemma_with_expert.paligemma.lm_head",
        ),
        (
            re.compile(r"\.paligemma_with_expert\.paligemma\.language_model\.model"),
            ".paligemma_with_expert.paligemma.model.language_model",
        ),
        (
            re.compile(r"\.paligemma_with_expert\.paligemma\.vision_tower"),
            ".paligemma_with_expert.paligemma.model.vision_tower",
        ),
        (
            re.compile(r"\.paligemma_with_expert\.paligemma\.multi_modal_projector"),
            ".paligemma_with_expert.paligemma.model.multi_modal_projector",
        ),
    ]

    for key, value in pi0_le.items():
        new_key = key
        for pattern, replacement in transformations:
            new_key = pattern.sub(replacement, new_key)
        transformed_dict[new_key] = value

    # Handle tied weights: lm_head.weight and embed_tokens.weight share memory
    lm_head_key = None
    embed_tokens_key = None

    for key in transformed_dict:
        if key.endswith(".paligemma_with_expert.paligemma.lm_head.weight"):
            lm_head_key = key
        elif key.endswith(".paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"):
            embed_tokens_key = key
        if lm_head_key and embed_tokens_key:
            break

    if lm_head_key and not embed_tokens_key:
        embed_tokens_key = lm_head_key.replace(
            ".lm_head.weight", ".model.language_model.embed_tokens.weight"
        )
        transformed_dict[embed_tokens_key] = transformed_dict[lm_head_key]
    elif embed_tokens_key and not lm_head_key:
        lm_head_key = embed_tokens_key.replace(
            ".model.language_model.embed_tokens.weight", ".lm_head.weight"
        )
        transformed_dict[lm_head_key] = transformed_dict[embed_tokens_key]

    return transformed_dict



def restore_pi0_from_lerobot_state_dict(pi0: Pi0, pi0_le: Mapping[str, jt.Array]):
    """
    Restore the weights of a Pi0 model from a Lerobot state dict.

    .. note ::
        The model is modified in-place!!

    :param pi0: The Pi0 model to restore.
    :param pi0_le: The Lerobot state dict to restore from.

    .. seealso ::
        https://github.com/huggingface/lerobot/blob/55198de096f46a8e0447a8795129dd9ee84c088c/src/lerobot/policies/pi0/conversion_scripts/convert_pi0_to_hf_lerobot.py
    
    """

    pi0_le_compat = lerobot_transform_state_dict(pi0_le)


    pi0.state_proj.kernel.value = pi0_le_compat["model.state_proj.weight"].transpose()
    pi0.state_proj.bias.value = pi0_le_compat["model.state_proj.bias"].transpose()

    pi0.action_in_proj.kernel.value = pi0_le_compat["model.action_in_proj.weight"].transpose()
    pi0.action_in_proj.bias.value = pi0_le_compat["model.action_in_proj.bias"].transpose()

    pi0.action_out_proj.kernel.value = pi0_le_compat["model.action_out_proj.weight"].transpose()
    pi0.action_out_proj.bias.value = pi0_le_compat["model.action_out_proj.bias"].transpose()

    pi0.action_time_mlp_in.kernel.value = pi0_le_compat["model.action_time_mlp_in.weight"].transpose()
    pi0.action_time_mlp_in.bias.value = pi0_le_compat["model.action_time_mlp_in.bias"].transpose()

    pi0.action_time_mlp_out.kernel.value = pi0_le_compat["model.action_time_mlp_out.weight"].transpose()
    pi0.action_time_mlp_out.bias.value = pi0_le_compat["model.action_time_mlp_out.bias"].transpose()


    path_le = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model"

    pi0.PaliGemma["img"].embedding["kernel"].value = einops.rearrange(
        pi0_le_compat[f"{path_le}.embeddings.patch_embedding.weight"], 
        "outC inC kH kW -> kH kW inC outC",
    )
    pi0.PaliGemma["img"].embedding["bias"].value = pi0_le_compat[f"{path_le}.embeddings.patch_embedding.bias"].transpose()

    pi0.PaliGemma["img"].pos_embedding.value = jax.numpy.reshape(
        pi0_le_compat[f"{path_le}.embeddings.position_embedding.weight"],
        shape=pi0.PaliGemma["img"].pos_embedding.value.shape,
    )


    path_le = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers"

    path = lambda x: x.PaliGemma["img"].Transformer["encoderblock"]

    for pi0_key, pi0le_key in [
        (lambda x: path(x)["LayerNorm_0"]["scale"], lambda x, idx: x[f"{path_le}.{idx}.layer_norm1.weight"]),
        (lambda x: path(x)["LayerNorm_0"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.layer_norm1.bias"]),

        (lambda x: path(x)["LayerNorm_1"]["scale"], lambda x, idx: x[f"{path_le}.{idx}.layer_norm2.weight"]),
        (lambda x: path(x)["LayerNorm_1"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.layer_norm2.bias"]),

        (lambda x: path(x)["MlpBlock_0"]["Dense_0"]["kernel"], lambda x, idx: x[f"{path_le}.{idx}.mlp.fc1.weight"]),
        (lambda x: path(x)["MlpBlock_0"]["Dense_0"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.mlp.fc1.bias"]),

        (lambda x: path(x)["MlpBlock_0"]["Dense_1"]["kernel"], lambda x, idx: x[f"{path_le}.{idx}.mlp.fc2.weight"]),
        (lambda x: path(x)["MlpBlock_0"]["Dense_1"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.mlp.fc2.bias"]),

        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["key"]["kernel"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.k_proj.weight"]),
        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["key"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.k_proj.bias"]),
        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["value"]["kernel"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.v_proj.weight"]),
        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["value"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.v_proj.bias"]),
        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["query"]["kernel"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.q_proj.weight"]),
        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["query"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.q_proj.bias"]),
        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["out"]["kernel"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.out_proj.weight"]),
        (lambda x: path(x)["MultiHeadDotProductAttention_0"]["out"]["bias"], lambda x, idx: x[f"{path_le}.{idx}.self_attn.out_proj.bias"]),
    ]:
        o = pi0_key(pi0)
        o.value = jax.numpy.stack([
            pi0le_key(pi0_le_compat, layer_idx)
            .transpose()
            .reshape(o.value.shape[1:])
            for layer_idx in range(o.value.shape[0])
        ])


    path_le = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm"
    path = lambda x: x.PaliGemma["img"].Transformer["encoder_norm"]
    o = path(pi0)["scale"]
    o.value = pi0_le_compat[f"{path_le}.weight"].transpose()
    o = path(pi0)["bias"]
    o.value = pi0_le_compat[f"{path_le}.bias"].transpose()


    path_le = "model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    path = lambda x: x.PaliGemma["img"].head
    o = path(pi0)["kernel"]
    o.value = pi0_le_compat[f"{path_le}.weight"].transpose()
    o = path(pi0)["bias"]
    o.value = pi0_le_compat[f"{path_le}.bias"].transpose()


    path_le = "model.paligemma_with_expert.paligemma.model.language_model"

    # TODO or? "paligemma.language_model.lm_head.weight"
    pi0.PaliGemma["llm"].embedder["input_embedding"].value = (
        pi0_le_compat[f"{path_le}.embed_tokens.weight"]
    )

    for path_le, expert_suffix in [
        ("model.paligemma_with_expert.paligemma.model.language_model", ""),
        ("model.paligemma_with_expert.gemma_expert.model", "_1"),
    ]:

        for pi0_key, pi0le_key in [
            (lambda x: x.PaliGemma["llm"].layers["attn"][f"q_einsum{expert_suffix}"]["w"], lambda x, idx: x[f"{path_le}.layers.{idx}.self_attn.q_proj.weight"]),
            (lambda x: x.PaliGemma["llm"].layers["attn"][f"attn_vec_einsum{expert_suffix}"]["w"], lambda x, idx: x[f"{path_le}.layers.{idx}.self_attn.o_proj.weight"]),
        ]:
            o = pi0_key(pi0)
            o.value = jax.numpy.stack([
                pi0le_key(pi0_le_compat, layer_idx)
                .transpose()
                .reshape(o.value.shape[1:])
                for layer_idx in range(o.value.shape[0])
            ])
        kv_einsum_w = pi0.PaliGemma["llm"].layers["attn"][f"kv_einsum{expert_suffix}"]["w"]
        kv_einsum_w.value = jax.numpy.stack([
            jax.numpy.stack([
                pi0_le_compat[f"{path_le}.layers.{layer_idx}.self_attn.{key}.weight"]
                .transpose()
                .reshape(kv_einsum_w.value.shape[2:])
                for key in ["k_proj", "v_proj"]
            ])
            for layer_idx in range(kv_einsum_w.value.shape[0])
        ])

        gating_einsum = pi0.PaliGemma["llm"].layers[f"mlp{expert_suffix}"]["gating_einsum"]
        gating_einsum.value = jax.numpy.stack([
            jax.numpy.stack([
                pi0_le_compat[f"{path_le}.layers.{layer_idx}.mlp.{key}.weight"]
                .transpose()
                .reshape(gating_einsum.shape[2:])
                for key in ["gate_proj", "up_proj"]
            ])
            for layer_idx in range(gating_einsum.shape[0])
        ])

        mlp_linear = pi0.PaliGemma["llm"].layers[f"mlp{expert_suffix}"]["linear"]
        mlp_linear.value = jax.numpy.stack([
            pi0_le_compat[f"{path_le}.layers.{layer_idx}.mlp.down_proj.weight"]
            .transpose()
            for layer_idx in range(mlp_linear.value.shape[0])
        ])

        pre_attn_norm_scale = pi0.PaliGemma["llm"].layers[f"pre_attention_norm{expert_suffix}"]["scale"]
        pre_attn_norm_scale.value = jax.numpy.stack([
            pi0_le_compat[f"{path_le}.layers.{layer_idx}.input_layernorm.weight"]
            .transpose()
            for layer_idx in range(pre_attn_norm_scale.value.shape[0])
        ])

        post_attn_norm_scale = pi0.PaliGemma["llm"].layers[f"pre_ffw_norm{expert_suffix}"]["scale"]
        post_attn_norm_scale.value = jax.numpy.stack([
            pi0_le_compat[f"{path_le}.layers.{layer_idx}.post_attention_layernorm.weight"]
            .transpose()
            for layer_idx in range(post_attn_norm_scale.value.shape[0])
        ])

        final_norm_scale = getattr(pi0.PaliGemma["llm"], f"final_norm{expert_suffix}")["scale"]
        final_norm_scale.value = (
            pi0_le_compat[f"{path_le}.norm.weight"]
            .transpose()
        )


from typing import Mapping

import flax.nnx
import safetensors

from ._todo_checkpoint import check_pytree_equality




class LazySafeTensorDict(Mapping):
    def __init__(self, f: safetensors.safe_open):
        self._f = f

    def __getitem__(self, key):
        return self._f.get_tensor(key)
    
    def __len__(self):
        return len(self._f.keys())
    
    def __iter__(self):
        return iter(self._f.keys())
    
    def __contains__(self, key):
        return key in self._f.keys()


class LerobotPi0Checkpoint:
    """
    TODO doc
    
    """

    @classmethod
    def restore(cls, resource_or_state: str | Mapping, verify: bool = True) -> Pi0:
        state = ...
        match resource_or_state:
            case str():
                state = LazySafeTensorDict(
                    safetensors.safe_open(resource_or_state, framework="flax")
                )
            case state if isinstance(state, Mapping):
                state = state
            case unknown:
                raise ValueError(f"Invalid resource or state: {unknown}")

        def make_abstract_pi0():
            # TODO custom configs!!!!!!
            return flax.nnx.eval_shape(lambda: Pi0(flax.nnx.Rngs(0)))

        pi0 = make_abstract_pi0()
        restore_pi0_from_lerobot_state_dict(pi0, state)

        if verify:
            abstract_pi0 = make_abstract_pi0()
            check_pytree_equality(
                expected=flax.nnx.state(abstract_pi0).to_pure_dict(), 
                got=flax.nnx.state(pi0).to_pure_dict(), 
                check_shapes=True, 
                check_dtypes=False,
            )

        return pi0
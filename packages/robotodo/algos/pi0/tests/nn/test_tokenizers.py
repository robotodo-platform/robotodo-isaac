"""
TODO doc

"""


import pytest
import numpy
import openpi.models.tokenizer

from robotodo.algos.pi0.nn.tokenizers import Pi0Tokenizer, Pi05Tokenizer


@pytest.mark.parametrize(
    "prompt", 
    [
        "do something",
        "1234 H_O_T_T_O_G_O",
    ],
)
def test_pi0tokenizer_openpi_compliance(prompt):
    openpi_tokenizer = openpi.models.tokenizer.PaligemmaTokenizer()
    tokens_openpi, masks_openpi = openpi_tokenizer.tokenize(prompt)
    tokens_openpi = tokens_openpi[masks_openpi]

    pi0_tokenizer = Pi0Tokenizer()
    [tokens] = pi0_tokenizer([prompt])

    numpy.testing.assert_array_equal(tokens_openpi, tokens)


@pytest.mark.parametrize(
    "prompt", 
    [
        "do something",
        "1234 H_O_T_T_O_G_O",
    ],
)
@pytest.mark.parametrize(
    "state", 
    [
        [1, 2, 3, 4],
    ],
)
def test_pi05tokenizer_openpi_compliance(prompt, state):
    openpi_tokenizer = openpi.models.tokenizer.PaligemmaTokenizer()
    tokens_openpi, masks_openpi = openpi_tokenizer.tokenize(prompt, state)
    tokens_openpi = tokens_openpi[masks_openpi]

    pi0_tokenizer = Pi05Tokenizer()
    [tokens] = pi0_tokenizer([prompt], [state])

    numpy.testing.assert_array_equal(tokens_openpi, tokens)

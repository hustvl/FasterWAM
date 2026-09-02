import pytest
import torch

from fasterwam.models.wan22.fastwam import FastWAM


class _Tokenizer:
    def __init__(self, ids: torch.Tensor, mask: torch.Tensor) -> None:
        self.ids = ids
        self.mask = mask

    def __call__(self, prompt, *, return_mask, add_special_tokens):
        assert return_mask and add_special_tokens
        return self.ids.clone(), self.mask.clone()


class _TextEncoder:
    def __init__(self, embeddings: torch.Tensor) -> None:
        self.embeddings = embeddings

    def __call__(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert ids.shape[:2] == mask.shape
        return self.embeddings.clone()


def _model(embeddings: torch.Tensor, mask: torch.Tensor) -> FastWAM:
    model = FastWAM.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.tokenizer = _Tokenizer(torch.zeros(mask.shape, dtype=torch.long), mask)
    model.text_encoder = _TextEncoder(embeddings)
    return model


def test_encode_prompt_masks_padding_without_erasing_valid_tokens():
    embeddings = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

    encoded, returned_mask = _model(embeddings, mask).encode_prompt(["a", "b"])

    assert torch.equal(returned_mask, mask)
    assert torch.equal(encoded[0, :2], embeddings[0, :2])
    assert torch.equal(encoded[1, :1], embeddings[1, :1])
    assert torch.equal(encoded[1, 2], embeddings[1, 2])
    assert torch.count_nonzero(encoded[0, 2:]) == 0
    assert torch.count_nonzero(encoded[1, 3:]) == 0


def test_encode_prompt_rejects_mismatched_encoder_sequence_length():
    mask = torch.ones(1, 3, dtype=torch.bool)
    embeddings = torch.zeros(1, 2, 4)

    with pytest.raises(ValueError, match="matching the token mask"):
        _model(embeddings, mask).encode_prompt("prompt")

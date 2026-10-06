import pytest
import torch

from taikonation.models.transformer import TaikoTransformer


def test_transformer_accepts_boolean_padding_mask():
    model = TaikoTransformer(
        vocab_size=16,
        num_genres=1,
        num_difficulties=5,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        audio_feature_size=80,
        max_sequence_length=8,
    )
    src = torch.randn(2, 8, 80)
    tgt = torch.randint(0, 16, (2, 8))
    genre = torch.zeros(2, dtype=torch.long)
    difficulty = torch.full((2,), 3, dtype=torch.long)
    padding_mask = torch.zeros(2, 8, dtype=torch.bool)
    padding_mask[:, -2:] = True

    output = model(src, tgt, genre, difficulty, padding_mask=padding_mask)
    assert output.shape == (2, 8, 16)


def test_gradient_checkpointing_supports_padding_mask_backward():
    model = TaikoTransformer(
        vocab_size=12,
        num_genres=1,
        num_difficulties=5,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        audio_feature_size=80,
        max_sequence_length=4,
        use_gradient_checkpointing=True,
    )
    model.train()
    src = torch.randn(1, 4, 80, requires_grad=True)
    tgt = torch.randint(0, 12, (1, 4))
    padding_mask = torch.tensor([[False, False, False, True]])

    output = model(
        src,
        tgt,
        torch.zeros(1, dtype=torch.long),
        torch.full((1,), 3, dtype=torch.long),
        padding_mask=padding_mask,
    )
    output.sum().backward()
    assert src.grad is not None
    assert torch.isfinite(src.grad).all()


def test_transformer_rejects_wrong_padding_shape():
    model = TaikoTransformer(
        vocab_size=8,
        num_genres=1,
        num_difficulties=5,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        audio_feature_size=80,
        max_sequence_length=4,
    )

    with pytest.raises(ValueError, match="padding_mask"):
        model(
            torch.randn(1, 4, 80),
            torch.zeros(1, 4, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        )

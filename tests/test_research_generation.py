from types import SimpleNamespace

import numpy as np
import torch

from taikonation.data.tokenization import TaikoTokenizer
from taikonation.generation.generator import generate_chart


class _DummyModel:
    def __init__(self, vocab_size, max_len=4):
        self.pos_encoder = SimpleNamespace(pe=torch.zeros(1, max_len, 1))
        self.vocab_size = vocab_size
        self.first_decoder_token = None

    def eval(self):
        return self

    def __call__(self, encoder_input, decoder_input, genre_id, difficulty_id):
        if self.first_decoder_token is None:
            self.first_decoder_token = int(decoder_input[0, 0])
        logits = torch.full(
            (1, decoder_input.shape[1], self.vocab_size),
            -20.0,
            device=decoder_input.device,
        )
        logits[:, -1, 5] = 20.0
        return logits


def test_generate_chart_uses_boundary_token_and_fills_entire_window():
    tokenizer = TaikoTokenizer()
    model = _DummyModel(tokenizer.vocab_size, max_len=4)
    initial = tokenizer.vocab["ka"]

    tokens = generate_chart(
        model,
        np.zeros((4, 80), dtype=np.float32),
        tokenizer,
        difficulty_id=3,
        config={"data": {"max_sequence_length": 999}},
        device=torch.device("cpu"),
        initial_token_id=initial,
    )

    assert model.first_decoder_token == initial
    assert len(tokens) == 4

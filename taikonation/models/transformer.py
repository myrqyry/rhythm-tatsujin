import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings, allowing the transformer
    to understand the order of the sequence.
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

import torch.utils.checkpoint as checkpoint

class TaikoTransformer(nn.Module):
    def __init__(self, vocab_size, num_genres, num_difficulties, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, audio_feature_size=80, max_sequence_length=512,
                 use_gradient_checkpointing=False):
        super(TaikoTransformer, self).__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.genre_embedding = nn.Embedding(num_genres, d_model)
        self.difficulty_embedding = nn.Embedding(num_difficulties, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_sequence_length)

        # Audio feature projection
        self.audio_input_projection = nn.Linear(audio_feature_size, d_model)

        # Custom transformer layers for checkpointing control
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Final output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._compiled = False

    def _generate_square_subsequent_mask(self, sz):
        # Boolean masks compose cleanly with boolean padding masks on current
        # PyTorch and avoid the deprecated mixed float/bool mask path.
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

    def _checkpoint_forward(self, module, *args, **kwargs):
        """Wrapper for gradient checkpointing"""
        if self.training and self.use_gradient_checkpointing:
            return checkpoint.checkpoint(module, *args, **kwargs, use_reentrant=False)
        return module(*args, **kwargs)

    def compile_model_if_needed(self):
        """Compile the model with torch.compile if not already compiled."""
        if not self._compiled and hasattr(torch, 'compile'):
            print("Compiling the model...")
            compiled_model = torch.compile(self, mode="default", dynamic=True)
            compiled_model._compiled = True
            return compiled_model
        return self

    def forward(self, src, tgt, genre_id, difficulty_id, padding_mask=None):
        # Input validation
        if src.dim() != 3:
            raise ValueError(f"Expected src to be 3D (batch,seq,feat); got shape {tuple(src.shape)}")
        if tgt.dim() != 2:
            raise ValueError(f"Expected tgt to be 2D (batch,seq); got shape {tuple(tgt.shape)}")
        if padding_mask is not None and padding_mask.shape != tgt.shape:
            raise ValueError(
                f"Expected padding_mask shape {tuple(tgt.shape)}; got {tuple(padding_mask.shape)}"
            )

        # Prepare encoder input
        src = self.audio_input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # Prepare decoder input with conditioning
        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        genre_embed = self.genre_embedding(genre_id).unsqueeze(1).expand(-1, tgt_embed.size(1), -1)
        difficulty_embed = self.difficulty_embedding(difficulty_id).unsqueeze(1).expand(-1, tgt_embed.size(1), -1)
        tgt_prepared = self.pos_encoder(tgt_embed + genre_embed + difficulty_embed)

        # Create causal mask
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        # Let PyTorch select the best scaled-dot-product attention backend for
        # the current device. Forcing FlashAttention breaks CPU fallback and
        # older CUDA GPUs such as Turing/sm75.
        memory = self._checkpoint_forward(
            self.transformer_encoder,
            src,
            src_key_padding_mask=padding_mask,
        )
        output = self._checkpoint_forward(
            self.transformer_decoder,
            tgt_prepared,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )

        return self.fc_out(output)

    def get_memory_stats(self):
        """Return memory usage statistics"""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        return {'message': 'CUDA not available'}

# labels: name::bert author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import LlamaModel, LlamaConfig
import torch

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    raise ValueError(
        "TurnkeyML does not support pretrained weights for LLaMA2 "
        "because it has special licensing terms. See for details: "
        "https://huggingface.co/docs/transformers/model_doc/llama2"
    )
else:
    config = LlamaConfig(
        hidden_size=4096,
        num_attention_heads=32,
        intermediate_size=4096 * 4,
        num_hidden_layers=32,
        use_cache=True,
    )
    model = LlamaModel(config)

inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
    "position_ids": [[0]],
    "past_key_value": (
        torch.ones(
            batch_size,
            config.num_attention_heads,
            max_seq_length - 1,
            model.self_attn.head_dim,
            dtype=torch.float,
        ),
        torch.ones(
            batch_size,
            config.num_attention_heads,
            max_seq_length - 1,
            model.self_attn.head_dim,
            dtype=torch.float,
        ),
    ),
}

# Call model
model(**inputs)

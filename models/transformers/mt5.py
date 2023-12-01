# labels: name::mt5 author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import MT5Model, AutoConfig
import torch

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = MT5Model.from_pretrained("google/mt5-small")
else:
    config = AutoConfig.from_pretrained("google/mt5-small")
    model = MT5Model(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)

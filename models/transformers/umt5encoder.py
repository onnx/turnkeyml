# labels: name::umt5encoder author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import UMT5EncoderModel, AutoConfig
import torch

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = UMT5EncoderModel.from_pretrained("google/umt5-small")
else:
    config = AutoConfig.from_pretrained("google/umt5-small")
    model = UMT5EncoderModel(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)

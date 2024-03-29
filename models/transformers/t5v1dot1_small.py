# labels: name::t5v1dot1_small author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import T5Model, AutoConfig
import torch

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = T5Model.from_pretrained("t5-v1_1-small")
else:
    config = AutoConfig.from_pretrained("t5-v1_1-small")
    model = T5Model(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)

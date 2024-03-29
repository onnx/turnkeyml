# labels: name::madlad400_7b author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import AutoModelForSeq2SeqLM, AutoConfig
import torch

torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-7b-mt")
else:
    config = AutoConfig.from_pretrained("google/madlad400-7b-mt")
    model = AutoModelForSeq2SeqLM.from_config(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)

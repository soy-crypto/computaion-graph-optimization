from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel


model_name = "google/bert_uncased_L-2_H-128_A-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, attn_implementation="eager", add_pooling_layer=False).eval()
model.config.return_dict = False  # The first output is last_hidden_state.

inputs = tokenizer("I like cats", padding="max_length", truncation=True, max_length=64, return_tensors="pt")
args = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

with torch.no_grad():
    expected = model.forward(*args)[0]
    emb = model.embeddings
    emb.register_buffer("position_ids", emb.position_ids, persistent=True)
    emb.register_buffer("token_type_ids", emb.token_type_ids, persistent=True)
    program = torch.export.export(model, args)
    exported_model = program.module()
    actual = exported_model.forward(*args)[0]

assert "embeddings.position_ids" in program.state_dict
assert expected.shape == (1, 64, 128)
torch.testing.assert_close(actual, expected)
print("Outputs match for this input.", tuple(actual.shape))

build = Path("build")
build.mkdir(exist_ok=True)
torch.export.save(program, build / "bert-tiny-encoder.pt2")
np.savez(
    build / "bert-tiny-encoder-reference.npz",
    input_ids=inputs["input_ids"].numpy(),
    attention_mask=inputs["attention_mask"].numpy(),
    token_type_ids=inputs["token_type_ids"].numpy(),
    expected_output=expected.numpy(),
)
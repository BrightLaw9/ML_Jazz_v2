from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from jazz_lora.checkpointing import (
    initialize_adapter_weights,
    load_adapter_for_inference,
    restore_checkpoint,
    save_checkpoint,
)
from jazz_lora.config import ModelConfig
from jazz_lora.modeling import (
    attach_lora,
    ensure_language_model_compatibility,
    trainable_parameters,
)


class TinyAttentionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(4, 4)
        self.to_k = torch.nn.Linear(4, 4)
        self.to_v = torch.nn.Linear(4, 4)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(4, 4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.to_out(self.to_q(value) + self.to_k(value) + self.to_v(value))


class TinyLanguageModel(torch.nn.Module):
    class Config:
        max_new_tokens = 2

    config = Config()

    def forward(self, inputs_embeds, attention_mask, output_hidden_states, return_dict):
        del attention_mask, output_hidden_states, return_dict
        return type("Output", (), {"hidden_states": (inputs_embeds + 1,)})()


class LoraCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(lora_rank=2, lora_alpha=2)

    def test_injection_and_checkpoint_round_trip(self) -> None:
        model = TinyAttentionModel().requires_grad_(False)
        attach_lora(model, self.config)
        parameters = trainable_parameters(model)
        self.assertTrue(parameters)
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        loss = model(torch.randn(2, 4)).square().mean()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = save_checkpoint(
                model, optimizer, temporary, global_step=1, epoch=0
            )
            self.assertTrue((checkpoint / "adapter_model.safetensors").is_file())
            self.assertEqual(restore_checkpoint(model, optimizer, checkpoint), (1, 0))

            phase_two = TinyAttentionModel().requires_grad_(False)
            attach_lora(phase_two, self.config)
            initialize_adapter_weights(phase_two, checkpoint)
            original_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
                if "lora_" in key
            }
            phase_two_state = {
                key: value.detach().cpu()
                for key, value in phase_two.state_dict().items()
                if "lora_" in key
            }
            self.assertEqual(original_state.keys(), phase_two_state.keys())
            for key in original_state:
                self.assertTrue(torch.equal(original_state[key], phase_two_state[key]))

            reloaded = TinyAttentionModel().requires_grad_(False)
            load_adapter_for_inference(reloaded, checkpoint, self.config)
            self.assertTrue(trainable_parameters(reloaded))

    def test_legacy_gpt2_model_compatibility(self) -> None:
        pipe = type("Pipe", (), {"language_model": TinyLanguageModel()})()
        self.assertTrue(ensure_language_model_compatibility(pipe))
        result = pipe.generate_language_model(
            inputs_embeds=torch.zeros(1, 3, 4),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=2,
        )
        self.assertEqual(tuple(result.shape), (1, 2, 4))
        self.assertTrue(torch.equal(result[:, 0], torch.ones(1, 4)))
        self.assertTrue(torch.equal(result[:, 1], torch.full((1, 4), 2.0)))


if __name__ == "__main__":
    unittest.main()

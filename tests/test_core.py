from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from caption_config import CAPTIONS_BY_FOLDER, DEFAULT_CAPTIONS
from jazz_lora.captions import caption_family, choose_caption
from jazz_lora.config import load_config
from jazz_lora.manifest import read_jsonl, write_jsonl


class CoreTests(unittest.TestCase):
    def test_configs_load(self) -> None:
        smoke = load_config("configs/cpu_smoke.json")
        full = load_config("configs/train.json")
        self.assertEqual(smoke.train.max_steps, 2)
        self.assertEqual(full.data.sample_rate, 16_000)
        self.assertEqual(full.mel.n_mels, 64)

    def test_instrument_specific_captions(self) -> None:
        trio = Path("training_samples/jazz_piano_trio/song.wav")
        sax = Path("training_samples/jazz_sax/song.wav")
        self.assertIs(caption_family(trio), CAPTIONS_BY_FOLDER["jazz_piano_trio"])
        self.assertIs(caption_family(sax), CAPTIONS_BY_FOLDER["jazz_sax"])
        self.assertIs(
            caption_family(Path("training_samples/uncategorized/song.wav")),
            DEFAULT_CAPTIONS,
        )
        self.assertEqual(choose_caption(trio, 7, 42), choose_caption(trio, 7, 42))

    def test_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            rows = [{"audio_path": "x.wav", "split": "train", "caption": "jazz"}]
            write_jsonl(path, rows)
            self.assertEqual(list(read_jsonl(path)), rows)


if __name__ == "__main__":
    unittest.main()

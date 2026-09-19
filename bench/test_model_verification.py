#!/usr/bin/env python3
"""CPU-only regression checks for supported model verification and selection."""
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class ModelVerificationTests(unittest.TestCase):
    def test_supported_and_invalid_heads(self):
        # Execute the upstream verifier block, including its shard-header checks.
        source = (REPO / "verify.sh").read_text(encoding="utf-8")
        block = source.split('$PY - "$MODEL" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        # Declared bits and stored bits are independent: metadata must not hide
        # a mismatched packed tensor, in either direction.
        cases = [(8, 8, True, True, 0), (4, 4, True, True, 0),
                 (8, 4, True, True, 1), (4, 8, True, True, 1),
                 (16, 4, True, True, 1), (None, 4, True, True, 1),
                 (4, 4, False, True, 1), (4, 4, True, False, 1)]
        for bits, stored_bits, packed, scale, expected in cases:
            with self.subTest(bits=bits, stored_bits=stored_bits, packed=packed, scale=scale), tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp)
                vocab, hidden = 16, 256
                groups = {"embed": {"targets": ["re:.*embed_tokens$"], "weights": {"num_bits": 8}}}
                if bits is not None:
                    groups["head"] = {"targets": ["re:.*lm_head$"], "weights": {"num_bits": bits}}
                config = {"text_config": {"vocab_size": vocab, "hidden_size": hidden},
                          "quantization_config": {"config_groups": groups}}
                (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
                head = "lm_head.weight_packed" if packed else "lm_head.weight"
                tensors = [("model.embed_tokens.weight_packed", "I32", [vocab, hidden // 4], 4),
                           (head, "I32", [vocab, hidden * stored_bits // 32], 4)]
                if scale:
                    tensors.append(("lm_head.weight_scale", "BF16", [vocab, hidden // 128], 2))
                header, offset = {}, 0
                for name, dtype, shape, itemsize in tensors:
                    end = offset + shape[0] * shape[1] * itemsize
                    header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
                    offset = end
                # The verifier reads headers only; no tensor payload is needed.
                encoded = json.dumps(header).encode("utf-8")
                encoded += b" " * (-len(encoded) % 8)
                (model / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
                weights = {name: "weights.safetensors" for name in header}
                (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}), encoding="utf-8")
                result = subprocess.run([sys.executable, "-", tmp], input=block, text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                # A crash must not masquerade as a successful rejection case.
                self.assertEqual(result.stderr, "")
                if expected:
                    self.assertIn("FAIL", result.stdout)

    def test_single_verifies_selected_model_without_changing_batch(self):
        base = "models/Qwen3.8-27B-W4A16-AutoRound"
        cases = [("single", False, "", base),
                 ("single", True, "", base + "-fast"),
                 ("single", True, "custom model", "custom model"),
                 ("batch", True, "", base)]
        for mode, fast, override, expected in cases:
            with self.subTest(mode=mode, fast=fast, override=override), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for directory in ("docker", "single-user", "batch", base):
                    (root / directory).mkdir(parents=True, exist_ok=True)
                if fast:
                    (root / (base + "-fast")).mkdir()
                # Relocate only the container root; execute the real entrypoint.
                entry = (REPO / "docker/entrypoint.sh").read_text(encoding="utf-8")
                entry = entry.replace("cd /app", 'cd "$PWD"').replace("REPO=/app", 'REPO="$PWD"')
                (root / "entrypoint.sh").write_text(entry, encoding="utf-8", newline="\n")
                helper = (REPO / "single-user/select_model.sh").read_text(encoding="utf-8")
                (root / "single-user/select_model.sh").write_text(helper, encoding="utf-8", newline="\n")
                (root / "verify.sh").write_text('printf "%s\\n" "${MODEL:-$PWD/' + base + '}" > verified\n', newline="\n")
                single = 'REPO="$PWD"\nsource "$REPO/single-user/select_model.sh"\nprintf "%s\\n" "$MODEL" > served\n'
                (root / "single-user/start_qwen.sh").write_text(single, newline="\n")
                (root / "batch/start_qwen.sh").write_text('printf "%s\\n" "${MODEL:-$PWD/' + base + '}" > served\n', newline="\n")
                env = dict(os.environ, MODEL=override, PREPARE="0", VERIFY="1")
                run = subprocess.run(["bash", "entrypoint.sh", mode], cwd=root, env=env, text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                verified = (root / "verified").read_text().strip()
                served = (root / "served").read_text().strip()
                self.assertEqual(verified, served)
                if override:
                    self.assertEqual(served, expected)
                else:
                    self.assertTrue(served.endswith("/" + expected), served)


if __name__ == "__main__":
    unittest.main()

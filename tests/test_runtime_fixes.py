"""Regression checks for observed launcher and capped UTF-8 mining failures."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from src.mine_negatives import clean_token_ids


class ByteTokenizer:
    """A real UTF-8 decoder with byte fragments matching the capped-token case."""

    pieces = {10: b"A", 11: b"\xe2\x80", 12: b"\x83", 13: b"\xe2\x80"}

    def decode(self, values, **_kwargs):
        return b"".join(self.pieces[value] for value in values).decode("utf-8", errors="replace")


class RuntimeFixTest(unittest.TestCase):
    def test_absolute_python_launch_finds_its_environment_ninja(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bin"
            binary.mkdir()
            ninja = binary / "ninja"
            ninja.write_text("#!/bin/sh\nexit 0\n")
            ninja.chmod(0o755)
            with (
                patch.object(sys, "executable", str(binary / "python")),
                patch.dict(
                    os.environ, {"PATH": "/usr/bin", "PYTHONPATH": "/stale/source"}, clear=True
                ),
            ):
                environment = run.execution_environment({"HF_HUB_OFFLINE": "1"})
            self.assertEqual(shutil.which("ninja", path=environment["PATH"]), str(ninja))
            self.assertNotIn("PYTHONPATH", environment)
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def test_length_cap_accepts_only_exact_terminal_utf8_prefix(self):
        tokenizer = ByteTokenizer()
        tokens = [10, 11, 12, 13]
        text = "A\u2003"
        self.assertEqual(clean_token_ids(tokenizer, tokens, text, truncated=True), tokens[:-1])
        with self.assertRaises(ValueError):
            clean_token_ids(tokenizer, tokens, text)
        with self.assertRaises(ValueError):
            clean_token_ids(tokenizer, tokens, "changed" + text, truncated=True)
        self.assertEqual(clean_token_ids(tokenizer, tokens[:-1], text), tokens[:-1])


if __name__ == "__main__":
    unittest.main()

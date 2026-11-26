"""跨 Windows 与 Unix 换行形式保持评测文本指纹稳定。"""

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation_hashing import sha256_text_file


class EvaluationHashingTests(unittest.TestCase):
    def test_lf_and_crlf_forms_have_the_same_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_bytes("问题一\n问题二\n".encode("utf-8"))
            lf_digest = sha256_text_file(path)
            path.write_bytes("问题一\r\n问题二\r\n".encode("utf-8"))
            self.assertEqual(sha256_text_file(path), lf_digest)

    def test_digest_matches_the_existing_crlf_frozen_hash_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_bytes("问题一\n问题二\n".encode("utf-8"))
            legacy_frozen_digest = hashlib.sha256(
                "问题一\r\n问题二\r\n".encode("utf-8")
            ).hexdigest()
            self.assertEqual(sha256_text_file(path), legacy_frozen_digest)

    def test_text_change_changes_the_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_bytes("问题一\n".encode("utf-8"))
            original = sha256_text_file(path)
            path.write_bytes("问题二\n".encode("utf-8"))
            self.assertNotEqual(sha256_text_file(path), original)


if __name__ == "__main__":
    unittest.main()

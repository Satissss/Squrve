import tempfile
import unittest
from pathlib import Path

from squrve_bmsql.upstream_adapter import load_official_classes, upstream_revision


class UpstreamAdapterTests(unittest.TestCase):
    def test_revision_reads_git_head_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git" / "refs" / "heads").mkdir(parents=True)
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (root / ".git" / "refs" / "heads" / "main").write_text(
                "0123456789abcdef\n", encoding="utf-8"
            )
            self.assertEqual(upstream_revision(root), "0123456789abcdef")

    def test_loader_rejects_non_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_official_classes(directory)


if __name__ == "__main__":
    unittest.main()

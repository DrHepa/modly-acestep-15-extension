"""Persistent storage, idempotence and real HTTP Range handling on a local server."""
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from acestep_modly.assets import _download, ensure_assets, require_assets, valid_file
from acestep_modly.constants import EXTENSION_ID
from acestep_modly.paths import checkpoint_dir, resolve_models_root, safe_child
from acestep_modly.vendor import sync_model_code


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # Windows runners may expose TEMP through an 8.3 alias (RUNNER~1).
        # Compare canonical paths, as the storage resolver intentionally returns.
        self.root = Path(self.temporary.name).resolve()

    def test_host_custom_settings_override_saved_root(self):
        ext = self.root / "custom-extensions" / EXTENSION_ID
        config = self.root / "config"
        config.mkdir()
        (config / "settings.json").write_text(json.dumps({"extensionsDir": str(ext.parent), "modelsDir": str(self.root / "models-new")}))
        state = self.root / "saved.json"
        state.write_text(json.dumps({"extension_root": str(ext), "models_root": str(self.root / "old")}))
        actual = resolve_models_root(root=ext, env={"MODLY_USER_DATA": str(config)}, state_file=state)
        self.assertEqual(actual, self.root / "models-new")

    def test_default_layout_and_explicit_models(self):
        ext = self.root / "extensions" / EXTENSION_ID
        no_state = self.root / "absent.json"
        (self.root / "settings.json").write_text("{}")
        self.assertEqual(resolve_models_root(root=ext, env={}, state_file=no_state), self.root / "models")
        self.assertEqual(resolve_models_root({"models_dir": str(self.root / "weights")}, root=ext, env={}, state_file=no_state), self.root / "weights")
        self.assertEqual(checkpoint_dir(self.root / "weights"), self.root / "weights" / EXTENSION_ID / "checkpoints")

    def test_unknown_root_and_unsafe_paths_fail(self):
        with self.assertRaises(ValueError):
            resolve_models_root(root=self.root / "unbound", env={}, state_file=self.root / "absent")
        with self.assertRaises(ValueError):
            resolve_models_root(root=self.root / "external-disk" / "extensions" / EXTENSION_ID, env={}, state_file=self.root / "absent")
        for relative in ("../x", "a/../../x", "/x", "a\\x", "C:/x", "a//x"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_child(self.root, relative)
        with self.assertRaises(ValueError):
            resolve_models_root({"models_dir": str(self.root / "extension" / "weights")}, root=self.root / "extension", env={})

    def test_links_are_not_followed(self):
        target = self.root / "other"
        target.mkdir()
        try:
            (self.root / "link").symlink_to(target, target_is_directory=True)
        except OSError:
            self.skipTest("Creating symlinks requires additional Windows privileges")
        with self.assertRaises(ValueError):
            safe_child(self.root, "link/model")

    def test_windows_reparse_attribute_on_python_311(self):
        target = self.root / "junction"
        target.mkdir()
        actual = Path.lstat
        class Reparse:
            st_file_attributes = 0x400
            st_mode = 0o40755
        with patch.object(Path, "lstat", lambda p: Reparse() if p == target else actual(p)):
            with self.assertRaises(ValueError):
                safe_child(self.root, "junction/model")

    def test_valid_git_blob_and_sha256(self):
        data = b"pinned asset\n"
        target = self.root / "asset"
        target.write_bytes(data)
        self.assertTrue(valid_file(target, {"size": len(data), "git_blob_sha1": hashlib.sha1(b"blob 13\0" + data).hexdigest()}))
        self.assertTrue(valid_file(target, {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}))
        target.write_bytes(b"corrupt data\n")
        self.assertFalse(valid_file(target, {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}))

    def test_update_reuses_weights_without_any_state_marker(self):
        data = b"tiny test asset"
        spec = {"path": "model/weights.dat", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        lock = {"repo_id": "ACE-Step/Ace-Step1.5", "revision": "test-only-revision", "files": [spec]}
        calls = []
        def download(url, target, metadata, log):
            calls.append(url)
            target.write_bytes(data)
        first = ensure_assets(self.root, lambda _: None, lock=lock, downloader=download)
        second = ensure_assets(self.root, lambda _: None, lock=lock, downloader=download)
        self.assertEqual((first, second), ({"reused": 0, "downloaded": 1}, {"reused": 1, "downloaded": 0}))
        self.assertEqual(len(calls), 1)
        (self.root / spec["path"]).write_bytes(b"corruption")
        ensure_assets(self.root, lambda _: None, lock=lock, downloader=download)
        self.assertEqual(len(calls), 2)

    def test_missing_assets_fail_offline(self):
        with self.assertRaisesRegex(RuntimeError, "Repair"):
            require_assets(self.root, lock={"files": [{"path": "missing", "size": 10, "sha256": "0" * 64}]})
        with patch("acestep_modly.vendor.VENDOR", self.root), self.assertRaisesRegex(RuntimeError, "Repair"):
            sync_model_code(self.root, verify_only=True)

    def test_resumes_real_http_and_verifies_hash(self):
        data = b"local-range-test" * 1000
        ranges = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                ranges.append(self.headers.get("Range"))
                start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(data) - start))
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
                self.end_headers()
                self.wfile.write(data[start:])
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            target = self.root / "asset"
            (self.root / "asset.part").write_bytes(data[:100])
            spec = {"path": "asset", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            _download(f"http://127.0.0.1:{server.server_port}/asset", target, spec, lambda _: None)
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(ranges, ["bytes=100-"])
            self.assertFalse((self.root / "asset.part").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()

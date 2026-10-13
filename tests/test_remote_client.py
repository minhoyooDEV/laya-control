import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from remote_fakes import LayaPocError, ServerError, Unavailable

from laya_control import remote
from laya_control.store import ControlError

FAKE_MODULE = """
DEFAULT_URL = "http://127.0.0.1:7650"
def rows_to_jsonl(rows):
    return "x"
class Client:
    def __init__(self, url=None, timeout=10.0):
        self.url, self.timeout = url, timeout
"""


class ClientLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        (self.home / "client").mkdir()
        (self.home / "client" / "laya_poc_client.py").write_text(FAKE_MODULE)
        sys.modules.pop("laya_poc_client", None)

    def tearDown(self):
        sys.modules.pop("laya_poc_client", None)
        self.tmp.cleanup()

    def test_loads_client_from_poc_home_with_long_timeout(self):
        with mock.patch.dict(os.environ, {"LAYA_POC_HOME": str(self.home)}):
            made = remote.make_remote()
        self.assertEqual(made.client.timeout, remote.CLIENT_TIMEOUT_S)
        self.assertEqual(made.rows_to_jsonl([]), "x")

    def test_missing_checkout_is_a_controlled_error(self):
        absent = {"LAYA_POC_HOME": str(self.home / "absent")}
        with mock.patch.dict(os.environ, absent), self.assertRaises(ControlError):
            remote.make_remote()

    def test_module_is_reloaded_when_home_changes(self):
        other = tempfile.TemporaryDirectory()
        try:
            (Path(other.name) / "client").mkdir()
            (Path(other.name) / "client" / "laya_poc_client.py").write_text(FAKE_MODULE.replace('"x"', '"y"'))
            with mock.patch.dict(os.environ, {"LAYA_POC_HOME": str(self.home)}):
                first = remote.make_remote()
            with mock.patch.dict(os.environ, {"LAYA_POC_HOME": other.name}):
                second = remote.make_remote()
            self.assertEqual((first.rows_to_jsonl([]), second.rows_to_jsonl([])), ("x", "y"))
        finally:
            other.cleanup()


class CallTests(unittest.TestCase):
    def test_server_error_hides_server_message(self):
        def boom():
            raise ServerError(409, "parity_failed", "contains private row text")

        with self.assertRaises(ControlError) as caught:
            remote.call(boom)
        self.assertIn("409 parity_failed", str(caught.exception))
        self.assertNotIn("private", str(caught.exception))

    def test_unavailable_and_base_errors_are_translated(self):
        for exc in (Unavailable("tunnel down"), LayaPocError("x")):
            with self.assertRaises(ControlError) as caught:
                remote.call(lambda e=exc: (_ for _ in ()).throw(e))
            self.assertIn("unavailable", str(caught.exception))

    def test_unrelated_errors_propagate(self):
        with self.assertRaises(ZeroDivisionError):
            remote.call(lambda: 1 / 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import socket
import sys
import threading
import types
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "gaze_tracking" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "semg_control" / "tools"))

try:
    import serial  # noqa: F401
except ModuleNotFoundError:
    sys.modules["serial"] = types.SimpleNamespace()

from robot_target_safety import prepare_approach_point
from semg_ros_bridge import determine_stable_intent, parse_intent
from target_pose_sender import send_target_pose


class IntegrationPureTests(unittest.TestCase):
    def test_approach_height_is_added(self):
        command = prepare_approach_point(
            np.asarray([0.30, 0.10, 0.02]),
            0.06,
        )

        np.testing.assert_allclose(
            command,
            np.asarray([0.30, 0.10, 0.08]),
        )

    def test_workspace_rejects_bad_target(self):
        with self.assertRaises(ValueError):
            prepare_approach_point(
                np.asarray([0.70, 0.00, 0.02]),
                0.06,
            )

    def test_semg_protocol_and_voting(self):
        history = deque(maxlen=5)

        for _ in range(4):
            history.append(parse_intent(b"SEMG,CLOSE\r\n"))

        history.append("REST")

        self.assertEqual(
            determine_stable_intent(history, 4),
            "CLOSE",
        )
        self.assertIsNone(parse_intent(b"bad,data\n"))

    def test_pose_sender_normalizes_and_transmits(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        received = []

        def receive_once():
            connection, _ = server.accept()

            with connection:
                received.append(connection.recv(4096))

            server.close()

        receiver = threading.Thread(target=receive_once)
        receiver.start()

        send_target_pose(
            x=0.30,
            y=0.10,
            z=0.08,
            qx=0.0,
            qy=2.0,
            qz=0.0,
            qw=2.0,
            host="127.0.0.1",
            port=port,
        )

        receiver.join(timeout=2.0)
        self.assertFalse(receiver.is_alive())

        payload = json.loads(received[0].decode("utf-8"))

        self.assertAlmostEqual(
            payload["orientation"]["y"],
            2 ** -0.5,
        )
        self.assertAlmostEqual(
            payload["orientation"]["w"],
            2 ** -0.5,
        )


if __name__ == "__main__":
    unittest.main()

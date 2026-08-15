import json
import math
import socket

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5055

def send_target_pose(
    x,
    y,
    z,
    qx,
    qy,
    qz,
    qw,
    frame_id="base_link",
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    timeout=2.0,
):
    values = [x, y, z, qx, qy, qz, qw]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Pose values must all be finite")

    if not frame_id:
        raise ValueError("frame_id cannot be empty")

    quaternion_norm = math.sqrt(
        qx * qx
        + qy * qy
        + qz * qz
        + qw * qw
    )

    if quaternion_norm < 1e-9:
        raise ValueError("Quaternion cannot have zero magnitude")

    payload = {
        "frame_id": frame_id,
        "position": {
            "x": x,
            "y": y,
            "z": z,
        },
        "orientation": {
            "x": qx/ quaternion_norm,
            "y": qy/ quaternion_norm,
            "z": qz/ quaternion_norm,
            "w": qw/ quaternion_norm,
        },
    }

    message = json.dumps(payload) + "\n"
    encoded_message = message.encode("utf-8")

    with socket.create_connection(
        (host, port),
        timeout=timeout,
    ) as connection:
        connection.sendall(encoded_message)


if __name__ == "__main__":
    send_target_pose(
        x=0.381,
        y=0.000,
        z=0.226,
        qx=0.017,
        qy=0.707,
        qz=0.017,
        qw=0.707,
    )

    print("Target pose sent successfully")
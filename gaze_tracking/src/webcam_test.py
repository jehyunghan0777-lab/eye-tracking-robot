from __future__ import annotations

import sys
import time

import cv2


WINDOW_NAME = "Eye-Tracking Robot - Webcame Test"


def open_camera() -> cv2.VideoCapture:
	"""Open the default Windows camera, with a backend fallback."""

	camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

	if camera.isOpened():
		print("Camera opened using the DirectShow backend.")
		return camera

	print("DirectShow did not open the camera. trying autmoatic selection.")
	camera.release()

	return cv2.VideoCapture(0)

def main() -> int:
	camera = open_camera()

	if not camera.isOpened():
		print(
			"Error: OpenCV could not open camera 0.\n"
			"Close Camera, Zoom, teams, Discord, and browser video-call tabs.\n"
			"Also check Windows camera privacy settings.",
			file=sys.stderr,
		)
		return 1

	#Requesting reasonable resolution. The camera may choose a nearby mode.
	camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

	print(f"Camera resolution: {width} x {height}")
	print("Press Q or Escape to exit.")

	previous_time = time.perf_counter()
	smoothed_fps = 0.0

	try:
		while True:
			frame_received, frame = camera.read()

			if not frame_received or frame is None:
				print("ERROR: Failed to receive a camera frame.", file=sys.stderr)
				return 1

			frame = cv2.flip(frame, 1)

			current_time = time.perf_counter()
			elapsed = current_time - previous_time
			previous_time = current_time

			if elapsed > 0:
				instantaneous_fps = 1.0 / elapsed

				if smoothed_fps == 0:
					smoothed_fps = instantaneous_fps
				else:
					smoothed_fps = (0.9 * smoothed_fps + 0.1 * instantaneous_fps)

			cv2.putText(
				frame,
				f"Webcam OK | FPS: {smoothed_fps:1f}",
				(20,40),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)
			
			cv2.imshow(WINDOW_NAME, frame)

			key = cv2.waitKey(1) & 0xFF

			if key == ord("q") or key == 27:
				break
	
	finally:
		camera.release()
		cv2.destroyAllWindows()

	print("Webcam released cleanly.")
	return 0

if __name__ == "__main__":
	raise SystemExit(main())


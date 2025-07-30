import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

# Setup
WIDTH, HEIGHT = 640, 480
switch_threshold = 15
stable_counter = 0
required_stable_frames = 5
current_camera = "center"

# Open two cams
cam_center = cv2.VideoCapture(1)  # External webcam (center)
cam_laptop = cv2.VideoCapture(0)  # Laptop's internal cam (left)

cam_center.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam_center.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam_laptop.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam_laptop.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float32)

# Virtual camera setup
with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=20) as virtual_cam:
    print(f'🎥 Virtual camera initialized: {virtual_cam.device}')

    while True:
        ret_center, frame_center = cam_center.read()
        ret_laptop, frame_laptop = cam_laptop.read()
        if not (ret_center and ret_laptop):
            print("Camera feed error")
            break

        frame_center = cv2.resize(frame_center, (WIDTH, HEIGHT))
        frame_laptop = cv2.resize(frame_laptop, (WIDTH, HEIGHT))

        rgb = cv2.cvtColor(frame_center, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        yaw = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            image_points = np.array([
                [landmarks[1].x * WIDTH, landmarks[1].y * HEIGHT],
                [landmarks[152].x * WIDTH, landmarks[152].y * HEIGHT],
                [landmarks[33].x * WIDTH, landmarks[33].y * HEIGHT],
                [landmarks[263].x * WIDTH, landmarks[263].y * HEIGHT],
                [landmarks[78].x * WIDTH, landmarks[78].y * HEIGHT],
                [landmarks[308].x * WIDTH, landmarks[308].y * HEIGHT],
            ], dtype=np.float32)

            focal_length = WIDTH
            center = (WIDTH / 2, HEIGHT / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw = angles[1]

            target = "center" if -switch_threshold <= yaw <= switch_threshold else "laptop"

            if target == current_camera:
                stable_counter = 0
            else:
                stable_counter += 1
                if stable_counter >= required_stable_frames:
                    current_camera = target
                    stable_counter = 0
                    print(f"[SWITCH] → {current_camera.upper()} cam")

        # Select the output feed
        output = frame_center if current_camera == "center" else frame_laptop
        label = f"Cam: {current_camera.upper()} | Yaw: {yaw:.2f}°" if yaw is not None else "Detecting..."
        cv2.putText(output, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Send to virtual camera
        virtual_cam.send(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        virtual_cam.sleep_until_next_frame()

        # Optional: Show for debugging
        cv2.imshow("Debug Feed", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cam_center.release()
cam_laptop.release()
cv2.destroyAllWindows()

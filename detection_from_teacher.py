from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import cv2

# =========================
# Video paths
# =========================
SOURCE_VIDEO_PATH = "/home/naaya/keypoint/soccer/data/newGame1.mp4"
TARGET_VIDEO_PATH = "/home/naaya/keypoint/soccer/out/yolo_detection_output.mp4"

# =========================
# Class IDs
# =========================
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# =========================
# Load trained YOLO model
# =========================
YOLO_MODEL_PATH = "/home/naaya/keypoint/soccer/model/yolo8x8x1000.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# =========================
# Annotators
# =========================
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex([
        "#00BFFF",  # Player / Goalkeeper
        "#FF0000",  # Referee
        "#FFD700"   # Other
    ]),
    thickness=2
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.ColorPalette.from_hex(["#FFD700"]),
    base=20,
    height=17
)

# =========================
# Video setup
# =========================
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

tracker = sv.ByteTrack()
tracker.reset()

with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as video_sink:

    for frame in tqdm(frame_generator, desc="Detecting objects"):

        # YOLO inference
        result = yolo_model.predict(frame)[0]

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()
        class_id = boxes.cls.cpu().numpy().astype(int)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # =========================
        # Separate detections
        # =========================
        ball_detection = detections[detections.class_id == BALL_ID]
        player_detections = detections[detections.class_id == PLAYER_ID]
        referee_detections = detections[detections.class_id == REFEREE_ID]

        # Optional: goalkeeper-ийг тоглогчтой хамт харуулах бол
        goalkeeper_detections = detections[detections.class_id == GOALKEEPER_ID]

        # Ball box-ийг бага зэрэг томруулах
        if ball_detection.xyxy.shape[0] > 0:
            ball_detection.xyxy = sv.pad_boxes(
                xyxy=ball_detection.xyxy,
                px=10
            )

        # Ball-оос бусад объект дээр tracking хийх
        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(
            threshold=0.5,
            class_agnostic=True
        )
        all_detections = tracker.update_with_detections(all_detections)

        # =========================
        # Annotate frame
        # =========================
        annotated_frame = frame.copy()

        # Player, referee, goalkeeper
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections
        )

        # Ball
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detection
        )

        # Save frame
        video_sink.write_frame(annotated_frame)

print("Detection finished!")
print(f"Saved video: {TARGET_VIDEO_PATH}")
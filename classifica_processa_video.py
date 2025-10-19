from pathlib import Path
import os
import csv
import cv2
import mediapipe as mp

def process_and_label_video(input_video_path, output_video_path, output_csv_path, window_name="label & crop"):
    """
    For each frame with detected hands, ask for a label, draw landmarks for guidance,
    crop the first detected hand, convert to grayscale (3-ch), and write to output video.
    Also saves a CSV with [frame_index, label] rows.
    """
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video: {input_video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_gray = None

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    labeled_rows = []  # [frame_idx, label]
    frame_idx = 0

    print("Press any key to label frames with detected hands. Press '!' to stop this video.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            # Draw landmarks on a preview frame
            preview = frame.copy()
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(preview, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow(window_name, preview)
            print(f"\rFrame {frame_idx}: ", end="")
            key = cv2.waitKey(0)
            label = chr(key) if key != -1 else ''
            if label == '!':
                print("\nStopping labeling for this video by user request.")
                break

            labeled_rows.append([frame_idx, label])
            print(f"\rFrame {frame_idx}: {label}")

            # Use only the first hand to keep 1 output frame per input frame
            hand_landmarks = result.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            offset = 20
            x_min = max(x_min - offset, 0)
            y_min = max(y_min - offset, 0)
            x_max = min(x_max + offset, w)
            y_max = min(y_max + offset, h)
            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size > 0:
                hand_crop_resized = cv2.resize(hand_crop, (256, 256))
                hand_crop_gray = cv2.cvtColor(hand_crop_resized, cv2.COLOR_BGR2GRAY)
                hand_crop_gray_3ch = cv2.cvtColor(hand_crop_gray, cv2.COLOR_GRAY2BGR)
                if out_gray is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    out_gray = cv2.VideoWriter(str(output_video_path), fourcc, fps, (256, 256))
                out_gray.write(hand_crop_gray_3ch)

    cap.release()
    if out_gray is not None:
        out_gray.release()
    cv2.destroyAllWindows()

    # Ensure output directory exists for CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        writer.writerows(labeled_rows)

def main():
    # Base directory for videos
    base_dir = Path("/mnt/d/videos_alfabeto")
    
    # Create output directory if it doesn't exist
    output_base_dir = Path("/mnt/d/videos_alfabeto_cropped")
    output_base_dir.mkdir(exist_ok=True)
    
    # Process only MP4 videos in the directory and its subdirectories
    for video_path in base_dir.rglob("*.mp4"):
        try:
            # Get the relative path from the base directory
            rel_path = video_path.relative_to(base_dir)
            
            # Create the corresponding output directory structure
            output_dir = output_base_dir / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Output filenames (always mp4)
            output_video_path = output_dir / f"{video_path.stem}.mp4"
            output_csv_path = output_dir / f"{video_path.stem}.csv"
            
            # Skip if the video file already exists
            if output_video_path.exists() and output_csv_path.exists():
                print(f"\nSkipping {video_path.name} - outputs already exist at {output_dir}")
                continue
            
            print(f"Processing {video_path} -> {output_video_path} and labels CSV")
            process_and_label_video(video_path, output_video_path, output_csv_path)
            
            print(f"Saved video: {output_video_path}")
            print(f"Saved labels: {output_csv_path}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

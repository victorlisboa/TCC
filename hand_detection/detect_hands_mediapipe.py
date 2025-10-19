import cv2
import mediapipe as mp
from pathlib import Path

def crop_hands(input_video_path, output_video_path):
    cap = cv2.VideoCapture(str(input_video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_gray = None
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
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
                        out_gray = cv2.VideoWriter(str(output_video_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (256, 256))
                    out_gray.write(hand_crop_gray_3ch)
    cap.release()
    if out_gray is not None:
        out_gray.release()
    cv2.destroyAllWindows()

def main():
    # Base directory for videos
    base_dir = Path("/mnt/d/videos_palavras")
    
    # Create output directory if it doesn't exist
    output_base_dir = Path("/mnt/d/videos_palavras_cropped")
    output_base_dir.mkdir(exist_ok=True)
    
    # Process all videos in the directory and its subdirectories
    for video_path in base_dir.rglob("*.mp4"):
        try:
            # Get the relative path from the base directory
            rel_path = video_path.relative_to(base_dir)
            
            # Create the corresponding output directory structure
            output_dir = output_base_dir / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename maintaining the same structure
            output_filename = output_dir / f"{video_path.stem}.mp4"
            
            # Skip if the video file already exists
            if output_filename.exists():
                print(f"\nSkipping {video_path.name} - cropped video already exists at {output_filename}")
                continue
            
            # Crop the video and save
            print(f"Cropping {output_filename}")
            crop_hands(video_path, output_filename)
            
            print(f"Video saved to: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

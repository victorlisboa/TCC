import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path

def process_video(video_path):
    print(f"\nProcessing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hand = mp_hands.Hands(max_num_hands=1)
    
    vetores_frames = []
    count_frames = 0
    acaba = False
    
    while not acaba:
        success, frame = cap.read()
        
        if success:
            # Corta a imagem pra pegar só a parte da direita
            frame = frame[:, 640:1280]
            
            RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hand.process(RGB_frame)
            if result.multi_hand_world_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                handLandmarks = result.multi_hand_world_landmarks[0]
                atual_point = []
                
                for i in range(0, 21):
                    atual_point.append(handLandmarks.landmark[i].x)
                    atual_point.append(handLandmarks.landmark[i].y)
                    atual_point.append(handLandmarks.landmark[i].z)
                
                vetores_frames.append(atual_point)
                count_frames += 1
                print(f"\rFrames coletados: {count_frames}", end="")
            
            cv2.imshow("capture image", frame)
            cv2.waitKey(1)
        else:
            acaba = True
            print("\nVideo concluido!")
    
    cap.release()
    return vetores_frames

def main():
    # Base directory for videos
    base_dir = Path("/mnt/d/LIBRAS-HC-RGBDS-2011/videos/")
    
    # Create output directory if it doesn't exist
    output_base_dir = Path("/mnt/d/LIBRAS-HC-RGBDS-2011/video_features")
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
            output_filename = output_dir / f"{video_path.stem}.py"
            
            # Skip if the feature file already exists
            if output_filename.exists():
                print(f"\nSkipping {video_path.name} - features already exist at {output_filename}")
                continue
            
            # Process the video
            vetores_frames = process_video(video_path)
            
            # Save the features
            with open(output_filename, "w", encoding="utf-8") as arquivo:
                arquivo.write(f"vetores_frames = {repr(vetores_frames)}\n")
            
            print(f"Features saved to: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

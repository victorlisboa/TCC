import os
import sys
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, jpeg_quality: int = 95) -> int:
    """Extract frames from a video and save them as JPEG images.

    Returns the number of frames written.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    written = 0
    success, frame = capture.read()
    while success:
        filename = output_dir / f"frame_{frame_index:06d}.jpg"
        # cv2.imwrite accepts params list [cv2.IMWRITE_JPEG_QUALITY, value]
        if cv2.imwrite(str(filename), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
            written += 1
        else:
            raise RuntimeError(f"Failed to write frame {frame_index} to {filename}")

        frame_index += 1
        success, frame = capture.read()

    capture.release()
    return written


def derive_output_dir_from_video(video_path: Path) -> Path:
    base_name = video_path.stem
    return video_path.parent / base_name


def main():
    for i in range(1, 11):
        VIDEO_PATH = Path(f"/mnt/d/videos_alfabeto_cropped/pedro/{i}.mp4")
        video_path = VIDEO_PATH
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        output_dir = derive_output_dir_from_video(video_path)
        if output_dir.exists():
            print(f'Diretório {output_dir} já existe. Pulando.')
            continue

        written = extract_frames(video_path=video_path, output_dir=output_dir, jpeg_quality=95)
        print(f"Saved {written} frames to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


from penalty_vision import PlayerDetector, VideoProcessor
from penalty_vision.detection.detection_utils import visualize_video_detection
from penalty_vision.video.frames import resize_frame

if __name__ == '__main__':
    output_dir = '/Users/zippo/PycharmProjects/PenaltyVision/PenaltyVision/data/frames'
    vp = VideoProcessor("/Users/zippo/PycharmProjects/PenaltyVision/PenaltyVision/data/video/penalty_001.mp4")
    frames = vp.extract_frames(0, 30)
    resized_frames = []
    for frame in frames:
        resized = resize_frame(frame, target_size=(320, 240))
        resized_frames.append(resized)

    # save_frames(resized_frames[:5], output_dir, prefix="resized_frames")
    # save_frames(frames[:5], output_dir, prefix="frame")
    pd = PlayerDetector(weights_dir='/Users/zippo/PycharmProjects/PenaltyVision/PenaltyVision/checkpoints/')
    visualize_video_detection(vp, pd, max_frames=50)

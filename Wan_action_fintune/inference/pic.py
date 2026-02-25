import cv2
import os

def save_first_frame(video_path, output_path='first_frame.jpg'):
    """
    使用 OpenCV 提取视频首帧并保存为图片
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError("无法读取视频帧")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    cv2.imwrite(output_path, frame)
    print(f"首帧已保存至: {output_path}")
    return frame

# 使用示例
save_first_frame('/opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/inference/examples/videos/clip_000015.mp4', '/opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/inference/examples/outputs/first_frame.jpg')
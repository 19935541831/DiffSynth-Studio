import cv2
cap = cv2.VideoCapture("/project/peilab/Puxin/Wan_action/data/robotwin_dataset_train_episode0/videos/clip_000132.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"{w}x{h}") 
import cv2
import numpy as np

# 打开视频文件
video_path = "your_video.mp4"  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 初始化变量
max_laplacian_var = 0
best_frame = None
best_frame_index = -1
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用拉普拉斯算子
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
    laplacian_var = np.var(laplacian)  # 计算拉普拉斯响应的方差

    # 更新最佳帧
    if laplacian_var > max_laplacian_var:
        max_laplacian_var = laplacian_var
        best_frame = frame.copy()
        best_frame_index = frame_index

    frame_index += 1

cap.release()

# 显示最佳帧
if best_frame is not None:
    print(f"选择的开始帧索引：{best_frame_index}")
    cv2.imshow("Best Frame", best_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未找到合适的帧")

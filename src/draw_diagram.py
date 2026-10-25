import cv2
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from src.calculator import get_angle_by_3_points
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult, PoseLandmarksConnections

# 取得 POSE_CONNECTIONS
# POSE_CONNECTIONS = PoseLandmarksConnections.POSE_CONNECTIONS
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])


def DrawKPT(landmarks, output_path):
    """
    用途:
        用 matplotlib 畫出 BlazePose 的 3D 骨架圖

    參數:
        landmarks: pose_world_landmarks[0] (3D key points)
        output_path: image 儲存位置

    輸出:
        KPT 圖
    """

    xs = [lm.x for lm in landmarks]
    ys = [-lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]
    vs = [lm.visibility for lm in landmarks]
    ps = [lm.presence for lm in landmarks]
    fig = plt.figure(figsize = (15, 10))
    ax = fig.add_axes([0.0, 0.1, 0.8, 0.8], projection = '3d')

    # 畫關鍵點
    ax.scatter(xs, zs, ys, color = 'black', s = 10)

    for i, (x, z, y) in enumerate(zip(xs, zs, ys)):
        ax.text(
            x, z, y,
            str(i),
            color = 'DarkRed',
            fontsize = 6
        )

    # 繪上關鍵點座標
    info_x = 0.4
    info_y_start = 1.4
    info_z = 0.0
    line_spacing = 0.1

    for i, (x, z, y, v, p) in enumerate(zip(xs, zs, ys, vs, ps)):
        info_y = info_y_start - i * line_spacing
        ax.text(
            info_x + 2.0, info_z, info_y,
            f"{i}: ({x:.2f}, {y:.2f}, {z:.2f}, {v:.2f}, {p:.2f})",
            color = 'DarkRed',
            fontsize = 10
        )

    
    # 繪上手臂(右 : 11, 13, 15 | 左 : 12, 14, 22), 
    # 手與軀幹(右 : 13, 11, 23 | 左 : 14, 12, 24), 
    # 腳和軀幹(右 : 11, 23, 25 | 左 : 12, 24, 26), 
    # 膝蓋(右 : 23, 25, 27 | 左 : 24, 26, 28)的關鍵點夾角
    info_x = 0.4
    info_y_start = 2.3
    info_z = 0.0
    line_spacing = 0.1
    r_elbow = get_angle_by_3_points([xs[11], ys[11], zs[11]], [xs[15], ys[15], zs[15]], [xs[13], ys[13], zs[13]])
    l_elbow = get_angle_by_3_points([xs[12], ys[12], zs[12]], [xs[22], ys[22], zs[22]], [xs[14], ys[14], zs[14]])
    r_shoulder = get_angle_by_3_points([xs[13], ys[13], zs[13]], [xs[23], ys[23], zs[23]], [xs[11], ys[11], zs[11]])
    l_shoulder = get_angle_by_3_points([xs[14], ys[14], zs[14]], [xs[24], ys[24], zs[24]], [xs[12], ys[12], zs[12]])
    r_hip = get_angle_by_3_points([xs[11], ys[11], zs[11]], [xs[25], ys[25], zs[25]], [xs[23], ys[23], zs[23]])
    l_hip = get_angle_by_3_points([xs[12], ys[12], zs[12]], [xs[26], ys[26], zs[26]], [xs[24], ys[24], zs[24]])
    r_knee = get_angle_by_3_points([xs[23], ys[23], zs[23]], [xs[27], ys[27], zs[27]], [xs[25], ys[25], zs[25]])
    l_knee = get_angle_by_3_points([xs[24], ys[24], zs[24]], [xs[28], ys[28], zs[28]], [xs[26], ys[26], zs[26]])
    info_y = info_y_start - 0 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"r_elbow: ({r_elbow:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )
    info_y = info_y_start - 1 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"l_elbow: ({l_elbow:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )
    info_y = info_y_start - 2 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"r_shoulder: ({r_shoulder:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )
    info_y = info_y_start - 3 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"l_shoulder: ({l_shoulder:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )
    info_y = info_y_start - 4 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"r_hip: ({r_hip:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )
    info_y = info_y_start - 5 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"l_hip: ({l_hip:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )
    info_y = info_y_start - 6 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"r_knee: ({r_knee:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )
    info_y = info_y_start - 7 * line_spacing
    ax.text(
        info_x + 2.0, info_z, info_y,
        f"l_knee: ({l_knee:.2f})",
        color = 'DarkRed',
        fontsize = 10
    )

    # 左、右骨架
    left = [
        # 頭部
        (0, 5), 
        (5, 8),

        # 手部
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),

        # 腰部、腳
        (12, 24),
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        (30, 32)
    ]

    right = [
        # 頭部
        (0, 2), 
        (2, 7),

        # 手部
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),

        # 腰部、腳
        (11, 23),
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        (29, 31)
    ]

    # 畫骨架連線
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        color = ''
        xline = [landmarks[start_idx].x, landmarks[end_idx].x]
        yline = [-landmarks[start_idx].y, -landmarks[end_idx].y]
        zline = [landmarks[start_idx].z, landmarks[end_idx].z]

        # 顏色：左半邊橘色、中間白色、右半邊藍色
        if connection in left:
            color = 'DarkOrange'
        elif connection in right:
            color = 'Aqua'
        else:
            color = 'white'

        ax.plot(xline, zline, yline, color = color)

    # 坐標範圍與標籤
    ax.tick_params(labelsize = 8, pad = 6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev = 15)  # 視角

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    # time.sleep(1) 


def DrawOverlay(original_image, landmarks, output_path):
    """
    用途:
        用 mediapipe 畫出 BlazePose 的 Overlay 圖

    參數:
        original_image: 原始圖片
        landmarks: pose_landmarks[0] (2D key points)
        output_path: image 儲存位置

    輸出:
        Overlay 圖
    """

    image = original_image.copy()
    height, width = image.shape[:2]

    # 畫關鍵點
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

    # 畫骨架線
    for start, end in POSE_CONNECTIONS:
        if start < len(landmarks) and end < len(landmarks):
            x0, y0 = int(landmarks[start].x * width), int(landmarks[start].y * height)
            x1, y1 = int(landmarks[end].x * width), int(landmarks[end].y * height)
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, image)


def WriteOutput(landmarks, isFirstWrite, frame, output_path):
    """
    用途:
        將 BlazePose 的關鍵點座標寫入 .txt 檔中

    參數:
        landmarks: pose_world_landmarks[0] (3D key points)
        isFirstWrite: 是否第一次寫入
        frame: 當前幀數
        output_path: image 儲存位置

    輸出:
        BlazePose 的關鍵點座標資訊
    """

    # 紀錄關鍵點座標 .txt 檔於 output_path(Analysis)
    mode = 'w' if isFirstWrite else 'a'
    with open(output_path, mode) as f:
        if isFirstWrite:
            f.write("BlazePose 3D Landmarks\n")
            f.write("Index, X, Y, Z, Visibility, Presence\n")
        f.write("---------------\n")
        f.write(f"Frame: {frame}\n")
        for i, landmark in enumerate(landmarks):
            f.write(f"{i}:({landmark.x}, {-landmark.y}, {landmark.z}, {landmark.visibility}, {landmark.presence})\n")
        f.write("---------------\n")
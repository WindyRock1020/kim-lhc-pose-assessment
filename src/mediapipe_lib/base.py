#!/usr/bin/env python
# coding: utf-8

# #### 套件

# In[1]:


from enum import Enum
import numpy as np
import cv2
import math

from typing import Self


# ##### MediaPipe

# In[2]:


import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)
from mediapipe.tasks.python.components.containers.landmark import (
    Landmark,
    NormalizedLandmark,
)


# ##### Custom

# In[3]:


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.calculator import (
        angles_to_lhc_label,
        get_angle_between_two_lines_position,
        get_angle_by_3_points,
        get_dist_between_points,
        get_triangle_gravity_position,
        label_list_to_label_list_rule,
        label_list_rule_to_label_list,
    )
    from src.value import (
        KPT_IDX_DICT,
        KPT_LIST,
        # Floor: 虛擬關鍵點與替代關節對照（由 hip 推導）
        FLOOR_KPT_IDX_DICT,
        FLOOR_KPT_NAME_DICT,
        FLOOR_KPT_LIST,
        FLOOR_POINT_Y,
        JOINT_NAME_DICT,
        JOINT_IDX_DICT,
        # Floor: 只替代指定關節（目前為 hip）
        FLOOR_JOINT_NAME_DICT,
        FLOOR_JOINT_IDX_DICT,
        LineConnections,
        blazepose_line,
    )
else:
    from ..calculator import (
        angles_to_lhc_label,
        get_angle_between_two_lines_position,
        get_angle_by_3_points,
        get_dist_between_points,
        get_triangle_gravity_position,
        label_list_to_label_list_rule,
        label_list_rule_to_label_list,
    )
    from ..value import (
        KPT_IDX_DICT,
        KPT_LIST,
        # Floor: 虛擬關鍵點與替代關節對照（由 hip 推導）
        FLOOR_KPT_IDX_DICT,
        FLOOR_KPT_NAME_DICT,
        FLOOR_KPT_LIST,
        FLOOR_POINT_Y,
        JOINT_NAME_DICT,
        JOINT_IDX_DICT,
        # Floor: 只替代指定關節（目前為 hip）
        FLOOR_JOINT_NAME_DICT,
        FLOOR_JOINT_IDX_DICT,
        LineConnections,
        blazepose_line,
    )


# #### 類別

# ##### 姿勢地標直播類別

# In[ ]:


class PoseLandmarkerLiveStream:
    """姿勢地標直播類別"""

    landmarker: PoseLandmarker = None  # 姿勢地標模型
    "姿勢地標模型"
    result = PoseLandmarkerResult([], [])  # 姿勢地標結果(實時更新)
    "姿勢地標結果(實時更新)"
    current_image: mp.Image = None
    "影像畫面(實時更新)"
    current_timestamp_ms: int = -1
    "時間戳(實時更新)"

    def __init__(self, model_path: str):
        """初始化姿勢模型物件

        Args:
            model_path (str): 模型路徑
        """

        def update_result(
            result: PoseLandmarkerResult,
            output_image: mp.Image,
            timestamp_ms: int,
        ) -> None:
            """更新偵測結果

            Args:
                result (PoseLandmarkerResult): 偵測結果
                output_image (mp.Image): 輸出圖片
                timestamp_ms (int): 時間戳
            """
            self.result = result
            self.current_image = output_image
            self.current_timestamp_ms = timestamp_ms

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.LIVE_STREAM,
            result_callback=update_result,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def detect_async(self, frame: cv2.typing.MatLike, timestamp_ms: int):
        """實時偵測

        Args:
            frame (cv2.typing.MatLike): 影像幀
            timestamp_ms (int): 時間戳
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def close(self):
        """關閉模型"""
        self.landmarker.close()


# ##### MediaPipe 姿勢物件結果

# In[ ]:


class PoseResult:
    """MediaPipe 姿勢物件結果"""

    result = PoseLandmarkerResult([], [])

    def __init__(self, result: PoseLandmarkerResult):
        self.result = result

    # 基礎函式

    def get_kpt_pos_by_index(
        self, idx: int, get_3d: bool = False
    ) -> list[float, float] | list[float, float, float]:
        """使用關鍵點索引值來取得關鍵點座標

        Args:
            idx (int): 索引值
            get_3d (bool, optional): 是否需要取得 3D 關鍵點. Defaults to False.

        Returns:
            list[float, float] | list[float, float, float]: 關鍵點座標
        """
        # Floor 虛擬點：用名稱導回同一入口，避免覆蓋原關鍵點
        if idx in FLOOR_KPT_IDX_DICT:
            return self.get_kpt_pos_by_name(FLOOR_KPT_IDX_DICT[idx], get_3d)

        position = None
        # 取得座標
        if not get_3d:
            kpt: Landmark = self.result.pose_landmarks[0][idx]
            position = [kpt.x, kpt.y]
        else:
            kpt: NormalizedLandmark = self.result.pose_world_landmarks[0][idx]
            position = [kpt.x, kpt.y, kpt.z]

        return position

    def get_kpt_pos_by_name(
        self, kpt_name: str, get_3d: bool = False
    ) -> list[float, float] | list[float, float, float]:
        """使用關鍵點名稱來取得關鍵點座標

        Args:
            kpt_name (str): 關鍵點名稱
            get_3d (bool, optional): 是否需要取得 3D 關鍵點. Defaults to False.

        Returns:
            list[float, float] | list[float, float, float]: 關鍵點座標
        """
        position = None

        # Floor 虛擬點：由 hip 推導座標
        if kpt_name in FLOOR_KPT_NAME_DICT:
            return self.get_floor_kpt_pos_by_name(kpt_name, get_3d)

        # 取得索引值
        if KPT_LIST.count(kpt_name) > 0:
            idx = KPT_LIST.index(kpt_name)
        else:
            return None

        # 取得座標
        if not get_3d:
            kpt: Landmark = self.result.pose_landmarks[0][idx]
            position = [kpt.x, kpt.y]
        else:
            kpt: NormalizedLandmark = self.result.pose_world_landmarks[0][idx]
            position = [kpt.x, kpt.y, kpt.z]

        return position

    def get_floor_kpt_pos_by_name(
        self, kpt_name: str, get_3d: bool = False
    ) -> list[float, float] | list[float, float, float]:
        """取得 floor 虛擬關鍵點座標(由 hip 推導)

        Args:
            kpt_name (str): 虛擬關鍵點名稱
            get_3d (bool, optional): 是否需要取得 3D 關鍵點. Defaults to False.

        Returns:
            list[float, float] | list[float, float, float]: 虛擬關鍵點座標
        """
        # 僅處理 floor 虛擬點
        if kpt_name not in FLOOR_KPT_LIST:
            return None

        # floor 點由對應 hip 推導（不覆蓋 hip）
        hip_name = "left_hip" if kpt_name == "left_floor" else "right_hip"
        hip_pos = self.get_kpt_pos_by_name(hip_name, get_3d)
        if hip_pos is None:
            return None

        if get_3d:
            return [hip_pos[0], FLOOR_POINT_Y, hip_pos[2]]

        return [hip_pos[0], FLOOR_POINT_Y]
    # 資料列表函式

    def get_all_kpt_positions(self, get_3d: bool = False) -> list[list[float, float]]:
        """取得全部關鍵點

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[list[float, float]]: 全部關鍵點列表
        """
        positions = []

        if not get_3d:
            for kpt in self.result.pose_landmarks[0]:
                kpt: Landmark
                positions.append([kpt.x, kpt.y])
        else:
            for kpt in self.result.pose_world_landmarks[0]:
                kpt: NormalizedLandmark
                positions.append([kpt.x, kpt.y, kpt.z])

        return positions


# ##### 姿勢結果分析器

# In[ ]:


class ResultAnalyzer:
    """姿勢結果分析器"""

    pose_result: PoseResult

    def __init__(self, pose_result: PoseResult):
        self.pose_result = pose_result

    # 基礎函式

    def get_center_position_by_2_hand(self, get_3d: bool = False) -> list[float]:
        """取得雙手的中心點(使用雙手腕計算)

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[float]: 雙手中心點座標
        """
        left_hand = np.array(self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d))
        right_hand = np.array(
            self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)
        )
        hand_center = (left_hand + right_hand) / 2
        return hand_center.tolist()

    def get_body_gravity_position(self, get_3d: bool = False) -> list[float]:
        """取得身體重心座標

        計算方式:
        1. 計算兩臀部中心座標
        2. 計算兩肩與臀部中心這三點的重心座標

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[float]: 身體重心座標
        """
        # 取得所需關鍵點座標
        left_shoulder = self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)
        right_shoulder = self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)
        left_hip = self.pose_result.get_kpt_pos_by_name("left_hip", get_3d)
        right_hip = self.pose_result.get_kpt_pos_by_name("right_hip", get_3d)

        # 計算臀部中心座標
        center_hip: list[float] = ((np.array(left_hip) + right_hip) / 2).tolist()
        # 計算兩肩與臀部中心的重心座標
        gravity_position = get_triangle_gravity_position(
            left_shoulder, right_shoulder, center_hip
        )

        return gravity_position

    def get_joint_angle_by_name(
        self,
        center_joint_name: str,
        get_3d: bool = False,
        use_floor_points: bool = False,
    ) -> float:
        """使用名稱取得關節角度

        Args:
            center_joint_name (str): 關節名稱
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            float: 關節角度
        """
        # 取得關節連接關鍵點名稱
        # Floor 模式只替代指定關節（目前為 hip），其餘走原邏輯
        if use_floor_points and center_joint_name in FLOOR_JOINT_NAME_DICT:
            j1, j2 = FLOOR_JOINT_NAME_DICT[center_joint_name]
        else:
            j1, j2 = JOINT_NAME_DICT[center_joint_name]

        center_pos = self.pose_result.get_kpt_pos_by_name(center_joint_name, get_3d)

        if center_pos is None:
            return 0.0

        pos1 = self.pose_result.get_kpt_pos_by_name(j1, get_3d)

        # 處理虛擬的 "ground" 關鍵點
        if j2 in ["left_ground", "right_ground"]:
            pos2 = center_pos.copy()
            pos2[1] -= 1.0  # 建立一個在臀部正下方的點來模擬垂直線
        else:
            pos2 = self.pose_result.get_kpt_pos_by_name(j2, get_3d)

        if pos1 is None or pos2 is None:
            return 0.0

        return get_angle_by_3_points(pos1, pos2, center_pos)

    def get_joint_angle_by_index(
        self,
        center_joint_idx: int,
        get_3d: bool = False,
        use_floor_points: bool = False,
    ) -> float:
        """使用索引值取得關節角度

        Args:
            center_joint_idx (int): 關節索引值
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            float: 關節角度
        """
        # 取得關節連接關鍵點索引值
        # Floor 模式只替代指定關節（目前為 hip），其餘走原邏輯
        if use_floor_points and center_joint_idx in FLOOR_JOINT_IDX_DICT:
            j1, j2 = FLOOR_JOINT_IDX_DICT[center_joint_idx]
        else:
            j1, j2 = JOINT_IDX_DICT[center_joint_idx]

        center_pos = self.pose_result.get_kpt_pos_by_index(center_joint_idx, get_3d)

        if center_pos is None:
            return 0.0

        pos1 = self.pose_result.get_kpt_pos_by_index(j1, get_3d)

        # 處理虛擬的 "ground" 關鍵點索引
        # Floor 虛擬點：由 hip 推導的 floor 座標
        if use_floor_points and j2 in FLOOR_KPT_IDX_DICT:
            pos2 = self.pose_result.get_kpt_pos_by_index(j2, get_3d)
        elif j2 in [100, 101]:
            pos2 = center_pos.copy()
            pos2[1] -= 1.0  # 建立一個在臀部正下方的點來模擬垂直線
        else:
            pos2 = self.pose_result.get_kpt_pos_by_index(j2, get_3d)

        if pos1 is None or pos2 is None:
            return 0.0

        return get_angle_by_3_points(pos1, pos2, center_pos)

    def get_line_positions(
        self, line_type: LineConnections = blazepose_line, get_3d: bool = False
    ) -> list[list, list, list]:
        """取得組合線條的多組兩點座標

        Args:
            line_type (LineConnections, optional): 線條連接種類. Defaults to blazepose_line.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[list, list, list]: 線條座標列表(分別存放左、中、右邊的線條座標資訊)
        """
        # 線條列表
        left, center, right = [], [], []

        for pt1, pt2 in line_type.left_kpt:
            pos = [
                self.pose_result.get_kpt_pos_by_name(pt1, get_3d),
                self.pose_result.get_kpt_pos_by_name(pt2, get_3d),
            ]
            left.append(pos)

        for pt1, pt2 in line_type.center_kpt:
            pos = [
                self.pose_result.get_kpt_pos_by_name(pt1, get_3d),
                self.pose_result.get_kpt_pos_by_name(pt2, get_3d),
            ]
            center.append(pos)

        for pt1, pt2 in line_type.right_kpt:
            pos = [
                self.pose_result.get_kpt_pos_by_name(pt1, get_3d),
                self.pose_result.get_kpt_pos_by_name(pt2, get_3d),
            ]
            right.append(pos)

        return [left, center, right]

    def get_pose_shoulder_hip_staggered_angle(self, get_3d: bool = False) -> float:
        """取得身體姿勢的肩臀交錯角度

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            float: 肩臀交錯角度
        """
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)

        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)

        shoulder_vec = np.array([shoulder_l, shoulder_r])
        hip_vec = np.array([hip_l, hip_r])

        return get_angle_between_two_lines_position(shoulder_vec, hip_vec)

    def get_pose_shoulder_hip_staggered_angle_xz(self) -> float:
        """取得身體姿勢的肩臀交錯角度(xz軸)

        Returns:
            float: 肩臀交錯角度
        """
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)

        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)

        shoulder_vec = np.array(
            [
                [shoulder_l[0], shoulder_l[2]],
                [shoulder_r[0], shoulder_r[2]],
            ]
        )
        hip_vec = np.array(
            [
                [hip_l[0], hip_l[2]],
                [hip_r[0], hip_r[2]],
            ]
        )

        return get_angle_between_two_lines_position(shoulder_vec, hip_vec)

    def get_pose_shoulder_hip_staggered_angle_xy(self) -> float:
        """取得身體姿勢的肩臀交錯角度(xy軸)

        Returns:
            float: 肩臀交錯角度
        """
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)

        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)

        shoulder_vec = np.array(
            [
                [shoulder_l[0], shoulder_l[1]],
                [shoulder_r[0], shoulder_r[1]],
            ]
        )
        hip_vec = np.array(
            [
                [hip_l[0], hip_l[1]],
                [hip_r[0], hip_r[1]],
            ]
        )

        return get_angle_between_two_lines_position(shoulder_vec, hip_vec)

    def get_pose_shoulder_hip_staggered_angle_yz(self) -> float:
        """取得身體姿勢的肩臀交錯角度(yz軸)

        Returns:
            float: 肩臀交錯角度
        """
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)

        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)

        shoulder_vec = np.array(
            [
                [shoulder_l[1], shoulder_l[2]],
                [shoulder_r[1], shoulder_r[2]],
            ]
        )
        hip_vec = np.array(
            [
                [hip_l[1], hip_l[2]],
                [hip_r[1], hip_r[2]],
            ]
        )

        return get_angle_between_two_lines_position(shoulder_vec, hip_vec)

    # 複合函式(有使用到基礎函式)

    def get_a_hand_to_gravity_dist(self, is_left: bool, get_3d: bool = False) -> float:
        """取得單手到身體重心間的距離

        Args:
            is_left (bool): 是否為左手
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            float: 單手到身體重心間的距離
        """
        hand_kpt_name = "left_wrist" if is_left else "right_wrist"  # 手腕關鍵點名稱
        # 手腕座標
        hand = np.array(self.pose_result.get_kpt_pos_by_name(hand_kpt_name, get_3d))
        gravity = np.array(self.get_body_gravity_position(get_3d))  # 重心座標
        dist = get_dist_between_points(hand, gravity)  # 手腕到重心距離
        return dist

    def get_two_hands_center_to_gravity_dist(self, get_3d: bool = False) -> float:
        """取得雙手中心到身體重心間的距離

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            float: 雙手中心和身體重心間的距離
        """
        hand = np.array(self.get_center_position_by_2_hand(get_3d))  # 雙手中心座標
        gravity = np.array(self.get_body_gravity_position(get_3d))  # 重心座標
        dist = get_dist_between_points(hand, gravity)  # 雙手中心到重心距離
        return dist

    def get_all_joint_angles_by_name(
        self, get_3d: bool = False, use_floor_points: bool = False
    ) -> dict:
        """使用名稱取得姿勢的所有關節角度

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            dict: 所有關節角度字典
        """
        all_angles = {}
        for center in JOINT_NAME_DICT.keys():
            angle = self.get_joint_angle_by_name(center, get_3d, use_floor_points)
            all_angles.update({center: angle})
        return all_angles

    def get_all_joint_angles_by_index(
        self, get_3d: bool = False, use_floor_points: bool = False
    ) -> dict:
        """使用索引值取得姿勢的所有關節角度

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            dict: 所有關節角度字典
        """
        all_angles = {}
        for center in JOINT_IDX_DICT.keys():
            angle = self.get_joint_angle_by_index(center, get_3d, use_floor_points)
            all_angles.update({center: angle})
        return all_angles

    def get_lhc_label(self, get_3d: bool = False, use_floor_points: bool = False) -> str:
        """取得 LHC 身體姿勢標籤

        標籤描述:
        A1: 站立
        A2: 搬運高處物品
        A3: 微彎腰
        A4: 彎腰
        A5: 蹲姿、跪姿、跪坐姿勢

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            str: LHC 身體姿勢標籤
        """
        angle_name_dict = self.get_all_joint_angles_by_name(
            get_3d, use_floor_points
        )
        return angles_to_lhc_label(angle_name_dict)

    # 身體姿勢額外加分項目的判斷

    def check_if_trunk_is_twisted_or_lateral_inclination(
        self, get_3d: bool = False
    ) -> bool:
        """檢查軀幹是否扭轉/側傾

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 軀幹是否扭轉/側傾
        """
        result = False

        # 軀幹扭轉
        if self.check_if_trunk_is_twisted(get_3d):
            result = True
        # 軀幹側傾
        elif self.check_if_trunk_is_lateral_inclination(get_3d):
            result = True

        return result

    def check_if_trunk_is_twisted(self, get_3d: bool = False) -> bool:
        """檢查軀幹是否扭轉

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 軀幹是否扭轉
        """
        TWISTED_ANGLE = 15
        return self.get_pose_shoulder_hip_staggered_angle_xz() > TWISTED_ANGLE

    def get_pose_shoulder_height(self, get_3d: bool = False) -> float:
        """取得肩膀高度差

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            tuple[float, float]: 左肩膀高度、右肩膀高度
        """
        # 取得關鍵點座標
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        if get_3d:
            shoulder_diff = np.linalg.norm(np.array(shoulder_l) - np.array(shoulder_r))
        else:
            shoulder_diff = np.linalg.norm(shoulder_l[1] - shoulder_r[1])

        return shoulder_diff

    def check_if_trunk_is_lateral_inclination(self, get_3d: bool = False) -> bool:
        """檢查軀幹是否側傾

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 軀幹是否側傾
        """
        HEIGHT_DIFF = 0.16  # 高度差

        # 取得關鍵點座標
        shoulder_l = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        shoulder_r = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)

        # 取得肩膀高度差
        shoulder_diff = np.linalg.norm(shoulder_l[1] - shoulder_r[1])

        return shoulder_diff > HEIGHT_DIFF

    def get_pose_gravity_distance(self, get_3d: bool = False) -> float:
        """取得身體重心在 3D 空間的距離

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            float: 身體重心在3D空間/xz軸平面的距離
        """
        if get_3d:
            wrist_l = self.pose_result.get_kpt_pos_by_name("left_wrist", True)
            wrist_r = self.pose_result.get_kpt_pos_by_name("right_wrist", True)
            hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)
            hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)
        else:
            wrist_l = self.pose_result.get_kpt_pos_by_name("left_wrist", True)[0::2]
            wrist_r = self.pose_result.get_kpt_pos_by_name("right_wrist", True)[0::2]
            hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)[0::2]
            hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)[0::2]
        
        gravity = np.mean([hip_l, hip_r], axis=0)
        # 左手到重心距離
        left_hand_to_gravity = np.linalg.norm(np.array(wrist_l) - gravity)
        # 右手到重心距離
        right_hand_to_gravity = np.linalg.norm(np.array(wrist_r) - gravity)
        dist = max(left_hand_to_gravity, right_hand_to_gravity)
        return dist

    def check_if_hands_at_a_distance(self, get_3d: bool = False) -> bool:
        """檢查手或重心是否遠離身體(只會算xz軸平面)

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 手或重心是否遠離身體
        """
        # 手遠離身體時，手到重心的距離
        DIST = 0.27

        result = False  # 判斷結果

        # 取得關鍵點座標
        wrist_l = self.pose_result.get_kpt_pos_by_name("left_wrist", True)[0::2]
        wrist_r = self.pose_result.get_kpt_pos_by_name("right_wrist", True)[0::2]
        hip_l = self.pose_result.get_kpt_pos_by_name("left_hip", True)[0::2]
        hip_r = self.pose_result.get_kpt_pos_by_name("right_hip", True)[0::2]

        gravity = np.mean([hip_l, hip_r], axis=0)

        # 左手到重心距離
        left_hand_to_gravity = np.linalg.norm(np.array(wrist_l) - gravity)
        # 右手到重心距離
        right_hand_to_gravity = np.linalg.norm(np.array(wrist_r) - gravity)

        # 左手到重心距離 大於 手遠離身體時，手到重心的距離
        left_result = False
        if left_hand_to_gravity > DIST:
            left_result = True

        # 右手到重心距離 大於 手遠離身體時，手到重心的距離
        right_result = False
        if right_hand_to_gravity > DIST:
            right_result = True

        # 只要其中一隻手符合，則都會被認定為真
        result = left_result or right_result

        return result

    def get_pose_coor_arm_Y(self) -> dict[str, float]:
        """取得肩膀、手肘、手腕的Y座標

        Returns:
            list[float]: 肩膀、手肘、手腕的Y座標
        """
        # 取得關鍵點座標
        left_shoulder = self.pose_result.get_kpt_pos_by_name("left_shoulder", True)
        left_elbow = self.pose_result.get_kpt_pos_by_name("left_elbow", True)
        left_wrist = self.pose_result.get_kpt_pos_by_name("left_wrist", True)

        right_shoulder = self.pose_result.get_kpt_pos_by_name("right_shoulder", True)
        right_elbow = self.pose_result.get_kpt_pos_by_name("right_elbow", True)
        right_wrist = self.pose_result.get_kpt_pos_by_name("right_wrist", True)
        return {
            "left_shoulder": -left_shoulder[1],
            "right_shoulder": -right_shoulder[1],
            "left_elbow": -left_elbow[1],
            "right_elbow": -right_elbow[1],
            "left_wrist": -left_wrist[1],
            "right_wrist": -right_wrist[1],
        }
    
    def check_if_arms_raised(self, get_3d: bool = False) -> bool:
        """檢查手臂是否需抬舉，手的水平位於手肘與肩膀之間

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 手臂是否需抬舉，手的水平位於手肘與肩膀之間
        """
        RAISED_ANGLE = 60  # 判斷抬舉的角度

        result = False  # 判斷結果

        # 左肩膀角度
        left_shoulder_angle = self.get_joint_angle_by_name("left_shoulder", get_3d)
        # 左肩膀的 y 軸位置
        left_shoulder_y = (
            self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)[1] * -1
        )
        # 左手肘的 y 軸位置
        left_elbow_y = (
            self.pose_result.get_kpt_pos_by_name("left_elbow", get_3d)[1] * -1
        )
        # 左手腕的 y 軸位置
        left_wrist_y = (
            self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)[1] * -1
        )
        left_result = False  # 表示左側是否達標

        # 右肩膀角度
        right_shoulder_angle = self.get_joint_angle_by_name("right_shoulder", get_3d)
        # 右肩膀的 y 軸位置
        right_shoulder_y = (
            self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)[1] * -1
        )
        # 右手肘的 y 軸位置
        right_elbow_y = (
            self.pose_result.get_kpt_pos_by_name("right_elbow", get_3d)[1] * -1
        )
        # 右手腕的 y 軸位置
        right_wrist_y = (
            self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)[1] * -1
        )
        right_result = False  # 表示左側是否達標

        ### (x) 在 y 軸中，當手在手肘和肩膀之間時，肩膀到手肘的長度 會大於 手肘到手腕的長度

        # 判斷左手抬舉行為，再判斷手的水平是否位於手肘和肩膀中間
        if left_shoulder_angle > RAISED_ANGLE:
            if left_shoulder_y > left_wrist_y and left_wrist_y > left_elbow_y:
                left_result = True

        # 判斷右手抬舉行為，再判斷手的水平是否位於手肘和肩膀中間
        if right_shoulder_angle > RAISED_ANGLE:
            if right_shoulder_y > right_wrist_y and right_wrist_y > right_elbow_y:
                right_result = True

        # 只要出現其中一種狀況，就表示為真
        result = left_result or right_result

        return result

    def check_if_hands_above_shoulder(self, get_3d: bool = False) -> bool:
        """檢查手是否會高過肩膀

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 手是否會高過肩膀
        """
        # 左手腕的 y 軸位置
        left_wrist_y = (
            self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)[1] * -1
        )
        # 左肩膀的 y 軸位置
        left_shoulder_y = (
            self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)[1] * -1
        )

        # 右手腕的 y 軸位置
        right_wrist_y = (
            self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)[1] * -1
        )
        # 右肩膀的 y 軸位置
        right_shoulder_y = (
            self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)[1] * -1
        )

        # 只要出現其中一種狀況(左手腕比左肩膀高 或 右手腕比右肩膀高)，就表示為真
        result = left_wrist_y > left_shoulder_y or right_wrist_y > right_shoulder_y

        return result

    # 測試區域

    def get_lhc_label_by_one_side(
        self,
        is_left: bool = False,
        get_3d: bool = False,
        use_floor_points: bool = False,
    ) -> str:
        """取得其中一邊的 LHC 身體姿勢標籤

        標籤描述:
        A1: 站立
        A2: 搬運高處物品
        A3: 微彎腰
        A4: 彎腰
        A5: 蹲姿、跪姿、跪坐姿勢

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            str: LHC 身體姿勢標籤
        """
        angle_name_dict = self.get_all_joint_angles_by_name(
            get_3d, use_floor_points
        )
        return angles_to_lhc_label(angle_name_dict)

    def check_if_hands_at_a_distance_by_one_side(
        self, is_left: bool = False, get_3d: bool = False
    ) -> bool:
        """檢查其中一邊的手或重心是否遠離身體

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 手或重心是否遠離身體
        """
        # 手遠離身體時，整個手臂長 和 手到重心距離 的比率(臂長: 手到重心距離 = 1: RATE)
        # 或是 手到重心距離 / 整個手臂長 = RATE
        # 或是 手到重心距離 = 整個手臂長 * RATE
        RATE = 0.8

        result = False  # 判斷結果

        if is_left:
            # 取得關鍵點座標
            left_wrist = self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)
            left_elbow = self.pose_result.get_kpt_pos_by_name("left_elbow", get_3d)
            left_shoulder = self.pose_result.get_kpt_pos_by_name(
                "left_shoulder", get_3d
            )
            # 左臂長度
            left_arm = get_dist_between_points(left_wrist, left_elbow, left_shoulder)
            # 左手到重心距離
            left_hand_to_gravity = self.get_a_hand_to_gravity_dist(True, get_3d)

            # 左手到重心距離 大於 左臂長度*比率
            if left_hand_to_gravity > left_arm * RATE:
                result = True
        else:
            # 取得關鍵點座標
            right_wrist = self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)
            right_elbow = self.pose_result.get_kpt_pos_by_name("right_elbow", get_3d)
            right_shoulder = self.pose_result.get_kpt_pos_by_name(
                "right_shoulder", get_3d
            )
            # 右臂長度
            right_arm = get_dist_between_points(
                right_wrist, right_elbow, right_shoulder
            )

            # 右手到重心距離
            right_hand_to_gravity = self.get_a_hand_to_gravity_dist(False, get_3d)

            # 右手到重心距離 大於 右臂長度*比率
            if right_hand_to_gravity > right_arm * RATE:
                result = True
            pass

        return result

    def check_if_arms_raised_by_one_side(
        self, is_left: bool = False, get_3d: bool = False
    ) -> bool:
        """檢查手臂是否需抬舉，手的水平位於手肘與肩膀之間

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 手臂是否需抬舉，手的水平位於手肘與肩膀之間
        """
        RAISED_ANGLE = 60  # 判斷抬舉的角度

        result = False  # 判斷結果

        if is_left:
            # 左肩膀角度
            left_shoulder_angle = self.get_joint_angle_by_name("left_shoulder", get_3d)
            # 左肩膀的 y 軸位置
            left_shoulder_y = (
                self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)[1] * -1
            )
            # 左手肘的 y 軸位置
            left_elbow_y = (
                self.pose_result.get_kpt_pos_by_name("left_elbow", get_3d)[1] * -1
            )
            # 左手腕的 y 軸位置
            left_wrist_y = (
                self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)[1] * -1
            )

            ### (x) 在 y 軸中，當手在手肘和肩膀之間時，肩膀到手肘的長度 會大於 手肘到手腕的長度
            # 判斷左手抬舉行為，再判斷手的水平是否位於手肘和肩膀中間
            if left_shoulder_angle > RAISED_ANGLE:
                if left_shoulder_y > left_wrist_y and left_wrist_y > left_elbow_y:
                    result = True
        else:
            # 右肩膀角度
            right_shoulder_angle = self.get_joint_angle_by_name(
                "right_shoulder", get_3d
            )
            # 右肩膀的 y 軸位置
            right_shoulder_y = (
                self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)[1] * -1
            )
            # 右手肘的 y 軸位置
            right_elbow_y = (
                self.pose_result.get_kpt_pos_by_name("right_elbow", get_3d)[1] * -1
            )
            # 右手腕的 y 軸位置
            right_wrist_y = (
                self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)[1] * -1
            )

            ### (x) 在 y 軸中，當手在手肘和肩膀之間時，肩膀到手肘的長度 會大於 手肘到手腕的長度
            # 判斷右手抬舉行為，再判斷手的水平是否位於手肘和肩膀中間
            if right_shoulder_angle > RAISED_ANGLE:
                if right_shoulder_y > right_wrist_y and right_wrist_y > right_elbow_y:
                    result = True

        return result

    def check_if_hands_above_shoulder_by_one_side(
        self, is_left: bool = False, get_3d: bool = False
    ) -> bool:
        """檢查手是否會高過肩膀

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            bool: 手是否會高過肩膀
        """
        result = False

        if is_left:
            # 左手腕的 y 軸位置
            left_wrist_y = (
                self.pose_result.get_kpt_pos_by_name("left_wrist", get_3d)[1] * -1
            )
            # 左肩膀的 y 軸位置
            left_shoulder_y = (
                self.pose_result.get_kpt_pos_by_name("left_shoulder", get_3d)[1] * -1
            )
            result = left_wrist_y > left_shoulder_y
        else:
            # 右手腕的 y 軸位置
            right_wrist_y = (
                self.pose_result.get_kpt_pos_by_name("right_wrist", get_3d)[1] * -1
            )
            # 右肩膀的 y 軸位置
            right_shoulder_y = (
                self.pose_result.get_kpt_pos_by_name("right_shoulder", get_3d)[1] * -1
            )
            result = right_wrist_y > right_shoulder_y

        return result

    pass


# ##### 頻率枚舉

# In[ ]:


class Frequency(Enum):
    """頻率枚舉"""

    RARELY = 0
    """幾乎沒有"""
    OCCASIONALLY = 1
    """偶爾"""
    FREQUENTLY_OR_CONSTANTLY = 2
    """通常"""


# ##### LHC 身體姿勢列表分析器

# In[ ]:


class LhcPoseListAnalyzer:
    """LHC 身體姿勢列表分析器"""

    result_lst: list[PoseLandmarkerResult] = []
    "姿勢分析結果列表"

    def __init__(self, result_lst: list[PoseLandmarkerResult] = None):
        if result_lst is not None:
            self.result_lst = result_lst

    # 特殊函式
    def __add__(self, other: PoseLandmarkerResult) -> list[PoseLandmarkerResult]:
        return self.result_lst + other

    def __iadd__(self, other: list[PoseLandmarkerResult]) -> Self:
        self.result_lst = self.result_lst + other
        return self

    def __getitem__(self, idx: int) -> PoseLandmarkerResult:
        return self.result_lst[idx]

    def __len__(self) -> int:
        return len(self.result_lst)

    def clean_data(self) -> None:
        "清除姿勢分析結果列表"
        self.result_lst = []

    # 基礎函式

    def get_frequency_from_bool_list(bool_lst: list[bool]) -> Frequency:
        """在 bool 列表中取得 True 出現的頻率

        Args:
            bool_lst (list[bool]): bool 列表

        Returns:
            Frequency: 頻率
        """
        # 計算 True 在 bool 列表中的比率
        rate = bool_lst.count(True) / len(bool_lst)
        # 進行頻率的分類
        frequency = Frequency.RARELY
        if rate > 1 / 3:
            frequency = Frequency.FREQUENTLY_OR_CONSTANTLY
        elif rate > 1 / 9:
            frequency = Frequency.OCCASIONALLY
        else:
            frequency = Frequency.RARELY
        return frequency

    def get_label_list_rule(self, get_3d: bool = False) -> list[list[str, int]]:
        """取得姿勢標籤列表的出現規則

        Args:
            get_3d (bool, optional): 姿勢標籤列表. Defaults to False.

        Returns:
            list[list[str, int]]: 姿勢標籤列表的出現規則
        """

        # 姿勢標籤列表
        labels = self.get_lhc_label_list(get_3d)
        # 姿勢標籤列表的出現規則
        result = label_list_to_label_list_rule(labels)

        return result

    def filter_label_list_in_fun_a(labels: list[str]) -> list[str]:
        """透過方法 A 來過濾標籤列表

        過濾方法:
            每 30 幀抓姿勢，設定其 30 幀中的最多姿勢為其代表姿勢

        Args:
            labels (list[str]): 原始標籤列表

        Returns:
            list[str]: 過濾後的標籤列表
        """
        STEP = 30  # 每 STEP 幀抓姿勢

        filtered_labels = []  # 過濾後的標籤列表

        # i 為 [0, STEP, STEP*2, STEP*3 , ...]
        for i in range(0, len(labels), STEP):
            labels_in_step = labels[i : i + STEP]  # 在 STEP 中的所有標籤

            # 拋棄最後不夠 STEP 幀影像的結果
            if len(labels_in_step) < STEP:
                break

            # 添加 STEP 數量的最多姿勢標籤
            values, counts = np.unique(labels_in_step, return_counts=True)
            filtered_labels += [str(values[counts.argmax()])] * STEP

        return filtered_labels

    def filter_label_list_in_fun_b(labels: list[str]) -> list[str]:
        """透過方法 B 來過濾標籤列表

        過濾方法:
            每 30 幀抓姿勢，觀察其 30 幀中的最多姿勢，並選擇其中有至少連續 5 幀的最多姿勢標籤為代表標籤

        Args:
            labels (list[str]): 原始標籤列表

        Returns:
            list[str]: 過濾後的標籤列表
        """
        STEP = 30  # 每 STEP 幀抓姿勢
        MIN_CONSECUTIVE_TIMES = 5  # 姿勢標籤最小連續次數

        filtered_labels = []  # 過濾後的標籤列表

        # i 為 [0, STEP, STEP*2, STEP*3 , ...]
        for i in range(0, len(labels), STEP):
            labels_in_step = labels[i : i + STEP]  # 在 STEP 中的所有標籤

            # 拋棄最後不夠 STEP 幀影像的姿勢標籤
            if len(labels_in_step) < STEP:
                break

            # 取得標籤列表規則
            rules = np.array(label_list_to_label_list_rule(labels_in_step))
            # 取得標籤出現的全部數值
            rules_num = rules[:, 1].astype(np.uint)
            # 取得標籤出現次數的最大數值
            max_num = rules_num.max()

            # 當姿勢最大出現次數小於所需最小連續次數，將計入空字串到輸出資料中
            if max_num < MIN_CONSECUTIVE_TIMES:
                filtered_labels += [""] * STEP
                continue

            # 取得標籤出現次數為最大數值的索引值
            max_idx = np.where(rules_num == max_num)
            # 取得最大述職的姿勢標籤
            max_label = rules[max_idx][0][0]
            # 添加姿勢標籤結果
            filtered_labels += [max_label] * STEP

        return filtered_labels

    def filter_label_list_in_fun_c(origin_labels: list[str]) -> list[str]:
        """透過方法 C 來過濾標籤列表

        過濾方法:
            每 30 幀抓姿勢，設定其 30 幀中的最多姿勢為其代表姿勢，最後不足 30 幀的影像會合併跟前一組一同計算

        Args:
            origin_labels (list[str]): 原始標籤列表

        Returns:
            list[str]: 過濾後的標籤列表
        """
        STEP = 30  # 每 STEP 幀抓姿勢(每秒影片約 30 幀)
        LEGAL_STEP = 10  # 合法的次數

        result = []  # 過濾後的標籤列表
        labels = np.array(origin_labels)  # np 格式的標籤列表

        if len(labels) < STEP * 2:
            # 直接計算最多次數的姿勢
            values, counts = np.unique(labels, return_counts=True)
            val_max = values[counts.argmax()]
            result = [val_max] * len(labels)
        else:
            filtered_labels = []  # 過濾後的標籤列表

            # 索引值列表(只是為了用在之後的標籤索引值)
            idxs = np.concatenate((range(0, len(labels), STEP), [len(labels)]))
            if idxs[-1] - idxs[-2] < STEP:
                idxs = np.delete(idxs, -2)

            for i in range(len(idxs) - 1):
                # 在 STEP 中的所有標籤
                labels_in_step = labels[idxs[i] : idxs[i + 1]]

                # 添加 STEP 數量的最多姿勢標籤
                values, counts = np.unique(labels_in_step, return_counts=True)
                filtered_labels += [values[counts.argmax()]] * len(labels_in_step)

            result = filtered_labels.copy()

        return result

    def filter_label_list_in_fun_d(origin_labels: list[str]) -> list[str]:
        """透過方法 D 來過濾標籤列表

        過濾方法:
            每 5 幀抓姿勢，設定其 5 幀中的最多姿勢為其代表姿勢，最後不足 5 幀的影像會合併跟前一組一同計算

        Args:
            origin_labels (list[str]): 原始標籤列表

        Returns:
            list[str]: 過濾後的標籤列表
        """
        STEP = 5  # 每 STEP 幀抓姿勢(每秒影片約 30 幀)
        LEGAL_STEP = 10  # 合法的次數

        result = []  # 過濾後的標籤列表
        labels = np.array(origin_labels)  # np 格式的標籤列表

        if len(labels) < STEP * 2:
            # 直接計算最多次數的姿勢
            values, counts = np.unique(labels, return_counts=True)
            val_max = values[counts.argmax()]
            result = [val_max] * len(labels)
        else:
            filtered_labels = []  # 過濾後的標籤列表

            # 索引值列表(只是為了用在之後的標籤索引值)
            idxs = np.concatenate((range(0, len(labels), STEP), [len(labels)]))
            if idxs[-1] - idxs[-2] < STEP:
                idxs = np.delete(idxs, -2)

            for i in range(len(idxs) - 1):
                # 在 STEP 中的所有標籤
                labels_in_step = labels[idxs[i] : idxs[i + 1]]

                # 添加 STEP 數量的最多姿勢標籤
                values, counts = np.unique(labels_in_step, return_counts=True)
                filtered_labels += [values[counts.argmax()]] * len(labels_in_step)

            result = filtered_labels.copy()

        return result

    # def filter_label_list_in_fun_e(origin_labels: list[str]) -> list[str]:
    #     """透過方法 E 來過濾標籤列表

    #     過濾方法:
    #         去除低於連續 15 幀的姿勢

    #     Args:
    #         origin_labels (list[str]): 原始標籤列表

    #     Returns:
    #         list[str]: 過濾後的標籤列表
    #     """

    #     smoothed_15 = smooth_label_lst(origin_labels)

    #     return smoothed_15

    # LHC 資料列表

    def get_lhc_label_list(
        self, get_3d: bool = False, use_floor_points: bool = False
    ) -> list[str]:
        """取得 LHC 標籤列表

        Args:
            get_3d (bool, optional): 是否取得 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            list[str]: LHC 標籤列表
        """
        return [
            ResultAnalyzer(PoseResult(result)).get_lhc_label(
                get_3d, use_floor_points
            )
            for result in self.result_lst
        ]

    def get_trunk_is_twisted_or_lateral_inclination_list(
        self, get_3d: bool = False
    ) -> list[bool]:
        """取得軀幹扭轉/側傾的 bool 列表

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 軀幹扭轉/側傾的 bool 列表
        """
        tmp_lst = [
            ResultAnalyzer(
                PoseResult(result)
            ).check_if_trunk_is_twisted_or_lateral_inclination(get_3d)
            for result in self.result_lst
        ]

        result = LhcPoseListAnalyzer.filter_label_list_in_fun_d(tmp_lst)
        return result

    def get_trunk_is_twisted_list(self, get_3d: bool = False) -> list[bool]:
        """取得軀幹扭轉的 bool 列表

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 軀幹扭轉的 bool 列表
        """
        tmp_lst = [
            ResultAnalyzer(PoseResult(result)).check_if_trunk_is_twisted(True)
            for result in self.result_lst
        ]

        result = LhcPoseListAnalyzer.filter_label_list_in_fun_d(tmp_lst)
        return result

    def get_trunk_is_lateral_inclination_list(self, get_3d: bool = False) -> list[bool]:
        """取得軀幹側傾的 bool 列表

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 軀幹側傾的 bool 列表
        """
        tmp_lst = [
            ResultAnalyzer(PoseResult(result)).check_if_trunk_is_lateral_inclination(
                True
            )
            for result in self.result_lst
        ]

        result = LhcPoseListAnalyzer.filter_label_list_in_fun_d(tmp_lst)
        return result

    def get_hands_at_a_distance_list(self, get_3d: bool = False) -> list[bool]:
        """取得手或重心遠離身體的 bool 列表

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 手或重心遠離身體的 bool 列表
        """
        tmp_lst = [
            ResultAnalyzer(PoseResult(result)).check_if_hands_at_a_distance(get_3d)
            for result in self.result_lst
        ]

        result = LhcPoseListAnalyzer.filter_label_list_in_fun_d(tmp_lst)
        return result

    def get_arms_raised_list(self, get_3d: bool = False) -> list[bool]:
        """取得手臂抬舉，手的水平位於手肘與肩膀之間的 bool 列表

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 手臂抬舉，手的水平位於手肘與肩膀之間的 bool 列表
        """
        tmp_lst = [
            ResultAnalyzer(PoseResult(result)).check_if_arms_raised(get_3d)
            for result in self.result_lst
        ]

        result = LhcPoseListAnalyzer.filter_label_list_in_fun_d(tmp_lst)
        return result

    def get_hands_above_shoulder_list(self, get_3d: bool = False) -> list[bool]:
        """取得手會高過肩膀的 bool 列表

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 手會高過肩膀的 bool 列表
        """
        tmp_lst = [
            ResultAnalyzer(PoseResult(result)).check_if_hands_above_shoulder(get_3d)
            for result in self.result_lst
        ]

        result = LhcPoseListAnalyzer.filter_label_list_in_fun_d(tmp_lst)
        return result

    # 取得身體姿勢額外加分項目的頻率

    def get_frequency_of_trunk_is_twisted(self, get_3d: bool = False) -> Frequency:
        """取得軀幹扭轉/側傾的頻率

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            Frequency: 頻率
        """
        check_list = self.get_trunk_is_twisted_or_lateral_inclination_list(
            get_3d
        )  # 取得檢查名單
        frequency = LhcPoseListAnalyzer.get_frequency_from_bool_list(
            check_list
        )  # 取得頻率
        return frequency

    def get_frequency_of_hands_at_a_distance(self, get_3d: bool = False) -> Frequency:
        """取得手或重心遠離身體的頻率

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            Frequency: 頻率
        """
        check_list = self.get_hands_at_a_distance_list(get_3d)  # 取得檢查名單
        frequency = LhcPoseListAnalyzer.get_frequency_from_bool_list(
            check_list
        )  # 取得頻率
        return frequency

    def get_frequency_of_arms_raised(self, get_3d: bool = False) -> Frequency:
        """取得手臂抬舉，手的水平位於手肘與肩膀之間的頻率

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            Frequency: 頻率
        """
        check_list = self.get_arms_raised_list(get_3d)  # 取得檢查名單
        frequency = LhcPoseListAnalyzer.get_frequency_from_bool_list(
            check_list
        )  # 取得頻率
        return frequency

    def get_frequency_of_hands_above_shoulder(self, get_3d: bool = False) -> Frequency:
        """取得手高過肩膀的頻率

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            Frequency: 頻率
        """
        check_list = self.get_hands_above_shoulder_list(get_3d)  # 取得檢查名單
        frequency = LhcPoseListAnalyzer.get_frequency_from_bool_list(
            check_list
        )  # 取得頻率
        return frequency

    # LHC 身體姿勢結果

    # def get_start_and_finish_poses(self, get_3d: bool = False) -> list[str, str]:
    #     """取得開始與結束姿勢的標籤

    #     Args:
    #         get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

    #     Returns:
    #         list[str, str]: 姿勢標籤[開始, 結束]
    #     """

    #     score = 0  # 姿勢評級分數
    #     start, end = "", ""  # 開始, 結束姿勢

    #     # 取得原始 LHC 姿勢標籤列表
    #     origin = self.get_lhc_label_list(get_3d)

    #     ### 平滑姿勢標籤列表
    #     # 移除低於 15 幀的連續姿勢
    #     smoothed_15 = smooth_label_lst(origin)
    #     # 移除低於 90 幀(3 秒)的連續姿勢
    #     smoothed_90 = smooth_label_lst(smoothed_15, 90)

    #     # 將字串去除抖動
    #     labels_15 = deduplicate_label_lst(smoothed_15)
    #     labels_90 = deduplicate_label_lst(smoothed_90)

    #     # 計算保持超過 3 秒的姿勢評級
    #     for lab in labels_90:
    #         tmp = calculate_posture_rating(lab, lab)  # 計算評級分數
    #         # 替代較低分的資訊
    #         if tmp >= score:
    #             score = tmp
    #             start, end = lab, lab

    #     # 計算身體有變化的姿勢評級
    #     start_end_labels = []
    #     if len(labels_15) > 1:
    #         start_end_labels = [
    #             [labels_15[i], labels_15[i + 1]] for i in range(len(labels_15) - 1)
    #         ]
    #     elif len(labels_15) == 1:
    #         start_end_labels = [[labels_15[0], labels_15[0]]]

    #     for _start, _end in start_end_labels:
    #         tmp = calculate_posture_rating(_start, _end)  # 計算評級分數
    #         # 替代較低分的資訊
    #         if tmp >= score:
    #             score = tmp
    #             start, end = _start, _end

    #     return [start, end]

    # def get_lhc_body_posture_rating_points(self, get_3d: bool = False) -> int:
    #     """取得 LHC 姿勢評級分數(不包含額外加分項)

    #     Args:
    #         get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

    #     Returns:
    #         int: 姿勢評級分數(不包含額外加分項)
    #     """
    #     start, end = self.get_start_and_finish_poses(get_3d)
    #     score = calculate_posture_rating(start, end)

    #     return score

    def get_lhc_body_posture_additional_points(self, get_3d: bool = False) -> float:
        """取得身體姿勢額外加分分數

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            float: 額外加分分數
        """
        score = -1.0  # 總分數
        extra_score_lst = np.array([0.0, 0.0, 0.0, 0.0])  # 額外分數列表
        # 頻率列表
        frequency_lst = [
            self.get_frequency_of_trunk_is_twisted(get_3d),
            self.get_frequency_of_hands_at_a_distance(get_3d),
            self.get_frequency_of_arms_raised(get_3d),
            self.get_frequency_of_hands_above_shoulder(get_3d),
        ]
        # 軀幹扭轉/側傾的分數
        if frequency_lst[0] is Frequency.FREQUENTLY_OR_CONSTANTLY:
            extra_score_lst[0] = 3.0
        elif frequency_lst[0] is Frequency.OCCASIONALLY:
            extra_score_lst[0] = 1.0
        # 手或重心遠離身體的分數
        if frequency_lst[1] is Frequency.FREQUENTLY_OR_CONSTANTLY:
            extra_score_lst[1] = 3.0
        elif frequency_lst[1] is Frequency.OCCASIONALLY:
            extra_score_lst[1] = 1.0
        # 手臂抬舉，手的水平位於手肘與肩膀之間的分數
        if frequency_lst[2] is Frequency.FREQUENTLY_OR_CONSTANTLY:
            extra_score_lst[2] = 1.0
        elif frequency_lst[2] is Frequency.OCCASIONALLY:
            extra_score_lst[2] = 0.5
        # 手高過肩膀的分數
        if frequency_lst[3] is Frequency.FREQUENTLY_OR_CONSTANTLY:
            extra_score_lst[3] = 2.0
        elif frequency_lst[3] is Frequency.OCCASIONALLY:
            extra_score_lst[3] = 1.0

        # 設定額外加分的總和分數
        score = extra_score_lst.sum()
        # 如果總和大於 6，分數則訂為6
        if score > 6.0:
            score = 6.0

        print(f"extra scores: {extra_score_lst}")
        print(f"sum of extra scores: {score}")

        return score

    def get_lhc_body_posture_total_points(self, get_3d: bool = False) -> float:
        """取得身體姿勢分數(包含姿勢評級、額外加分)

        Args:
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            float: 身體姿勢分數(包含姿勢評級、額外加分)
        """
        bp_rating_points = self.get_lhc_body_posture_rating_points(get_3d)
        addtional_points = self.get_lhc_body_posture_additional_points(get_3d)
        return bp_rating_points + addtional_points

    # 用於測試

    def get_lhc_label_list_by_one_side(
        self,
        is_left: bool = False,
        get_3d: bool = False,
        use_floor_points: bool = False,
    ) -> list[str]:
        """取得一邊的 LHC 標籤列表

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否取得 3D 姿勢. Defaults to False.
            use_floor_points (bool, optional): 是否使用 floor 參考點. Defaults to False.

        Returns:
            list[str]: LHC 標籤列表
        """
        return [
            ResultAnalyzer(PoseResult(result)).get_lhc_label_by_one_side(
                is_left, get_3d, use_floor_points
            )
            for result in self.result_lst
        ]

    def get_hands_at_a_distance_list_by_one_side(
        self, is_left: bool = False, get_3d: bool = False
    ) -> list[bool]:
        """取得一邊的手或重心遠離身體的 bool 列表

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 手或重心遠離身體的 bool 列表
        """
        return [
            ResultAnalyzer(PoseResult(result)).check_if_hands_at_a_distance_by_one_side(
                is_left, get_3d
            )
            for result in self.result_lst
        ]

    def get_arms_raised_list_by_one_side(
        self, is_left: bool = False, get_3d: bool = False
    ) -> list[bool]:
        """取得一邊的手臂抬舉，手的水平位於手肘與肩膀之間的 bool 列表

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 手臂抬舉，手的水平位於手肘與肩膀之間的 bool 列表
        """
        return [
            ResultAnalyzer(PoseResult(result)).check_if_arms_raised_by_one_side(
                is_left, get_3d
            )
            for result in self.result_lst
        ]

    def get_hands_above_shoulder_list_by_one_side(
        self, is_left: bool = False, get_3d: bool = False
    ) -> list[bool]:
        """取得一邊的手會高過肩膀的 bool 列表

        Args:
            is_left (bool, optional): 是否為左側姿勢. Defaults to False.
            get_3d (bool, optional): 是否為 3D 姿勢. Defaults to False.

        Returns:
            list[bool]: 手會高過肩膀的 bool 列表
        """
        return [
            ResultAnalyzer(
                PoseResult(result)
            ).check_if_hands_above_shoulder_by_one_side(is_left, get_3d)
            for result in self.result_lst
        ]

    pass

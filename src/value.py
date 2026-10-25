#!/usr/bin/env python
# coding: utf-8

# 套件

# In[ ]:


import numpy as np


# 線條連接點類別

# In[ ]:


class LineConnections:
    """線條連接點"""

    left_kpt: list[list[str]] = []  # 左邊關鍵點連接列表
    right_kpt: list[list[str]] = []  # 右邊關鍵點連接列表
    center_kpt: list[list[str]] = []  # 中間關鍵點連接列表
    full_kpt: list[list[str]] = []  # 全關鍵點連接列表

    def __init__(
        self, left: list[list[str]], right: list[list[str]], center: list[list[str]]
    ) -> None:
        """初始化物件
        Args:
            left (list[list[str]]): 左邊連接列表
            right (list[list[str]]): 右邊連接列表
            center (list[list[str]]): 中間連接列表
        """
        self.left_kpt = left
        self.right_kpt = right
        self.center_kpt = center
        lr_kpt = np.append(left, right, axis=0)
        full_kpt = np.append(lr_kpt, center, axis=0)
        self.full_kpt = full_kpt.tolist()


# BlazePose 線條組合列表

# In[ ]:


blazepose_line = LineConnections(
    left=[
        # 頭部
        ["nose", "left_eye"],  # 0, 2
        ["left_eye", "left_ear"],  # 2, 7
        # 手部
        ["left_shoulder", "left_elbow"],  # 11, 13
        ["left_elbow", "left_wrist"],  # 13, 15
        ["left_wrist", "left_pinky"],  # 15, 17
        ["left_wrist", "left_index"],  # 15, 19
        ["left_wrist", "left_thumb"],  # 15, 21
        ["left_pinky", "left_index"],  # 17, 19
        # 腰部、腳
        ["left_shoulder", "left_hip"],  # 11, 23
        ["left_hip", "left_knee"],  # 23, 25
        ["left_knee", "left_ankle"],  # 25, 27
        ["left_ankle", "left_heel"],  # 27, 29
        ["left_ankle", "left_foot_index"],  # 27, 31
        ["left_heel", "left_foot_index"],  # 29, 31
    ],
    right=[
        # 頭部
        ["nose", "right_eye"],  # 0, 5
        ["right_eye", "right_ear"],  # 5, 8
        # 手部
        ["right_shoulder", "right_elbow"],  # 12, 14
        ["right_elbow", "right_wrist"],  # 14, 16
        ["right_wrist", "right_pinky"],  # 16, 18
        ["right_wrist", "right_index"],  # 16, 20
        ["right_wrist", "right_thumb"],  # 16, 22
        ["right_pinky", "right_index"],  # 18, 20
        # 腰部、腳
        ["right_shoulder", "right_hip"],  # 12, 24
        ["right_hip", "right_knee"],  # 24, 26
        ["right_knee", "right_ankle"],  # 26, 28
        ["right_ankle", "right_heel"],  # 28, 30
        ["right_ankle", "right_foot_index"],  # 28, 32
        ["right_heel", "right_foot_index"],  # 30, 32
    ],
    center=[
        ["mouth_left", "mouth_right"],  # 9, 10
        ["left_shoulder", "right_shoulder"],  # 11, 12
        ["left_hip", "right_hip"],  # 23, 24
    ],
)


# 關鍵點索引值字典

# In[ ]:


KPT_IDX_DICT: dict = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


# 關鍵點列表

# In[ ]:


KPT_LIST: list = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


# 額外虛擬關鍵點(Floor)

# In[ ]:


FLOOR_POINT_Y: float = 10.0

FLOOR_KPT_IDX_DICT: dict = {
    100: "left_floor",
    200: "right_floor",
}

FLOOR_KPT_NAME_DICT: dict = {
    "left_floor": 100,
    "right_floor": 200,
}

FLOOR_KPT_LIST: list = [
    "left_floor",
    "right_floor",
]


# 關節角度名稱對照字典

# In[ ]:


JOINT_NAME_DICT: dict = {
    "left_shoulder": ["left_elbow", "left_hip"],
    "right_shoulder": ["right_elbow", "right_hip"],
    "left_hip": ["left_shoulder", "left_knee"],
    "right_hip": ["right_shoulder", "right_knee"],
    "left_knee": ["left_hip", "left_ankle"],
    "right_knee": ["right_hip", "right_ankle"],
}


# Floor 版本關節角度名稱對照字典(僅膝蓋改用 floor 參考點)

# In[ ]:


FLOOR_JOINT_NAME_DICT: dict = {
    "left_hip": ["left_shoulder", "left_floor"],
    "right_hip": ["right_shoulder", "right_floor"],
}


# 關節角度索引值對照字典

# In[ ]:


JOINT_IDX_DICT: dict = {
    11: [13, 23],
    12: [14, 24],
    23: [11, 25],
    24: [12, 26],
    25: [23, 27],
    26: [24, 28],
}


# Floor 版本關節角度索引值對照字典(僅膝蓋改用 floor 參考點)

# In[ ]:


FLOOR_JOINT_IDX_DICT: dict = {
    23: [11, 100],
    24: [12, 200],
}

CLEAN_JOINT_SOURCE: dict = {
    "LWrist": 15,
    "RWrist": 16,
    "LElbow": 13,
    "RElbow": 14,
    "LShoulder": 11,
    "RShoulder": 12,
    "LHip": 23,
    "RHip": 24,
    "LKnee": 25,
    "RKnee": 26,
    "LAnkle": 27,
    "RAnkle": 28,
    # "LHeel": 29,
    # "RHeel": 30,
}

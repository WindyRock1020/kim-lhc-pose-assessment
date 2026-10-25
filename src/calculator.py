#!/usr/bin/env python
# coding: utf-8

# #### 使用套件

# In[1]:


import numpy as np
import math


# #### 函式

# ##### 從三個座標點之間取得角度

# In[ ]:


def get_angle_by_3_points(pos1: list, pos2: list, center_pos: list) -> float:
    """
    從三個座標點之間取得角度
    中心點為 center_pos
    (使用餘弦定理)

    Args:
        pos1 (list): 座標 1
        pos2 (list): 座標 2
        center_pos (list): 中心關鍵點(連接關鍵點 1和關鍵點 2)

    Returns:
        float: 角度(度數)
    """
    # 求角 A (對面為邊 a)
    # a: pt1 pt2 距離
    # b: pt1 cenPt 距離
    # c: pt2 cenPt 距離

    p1 = np.array(pos1)
    p2 = np.array(pos2)
    center_pt = np.array(center_pos)

    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p1 - center_pt)
    c = np.linalg.norm(p2 - center_pt)

    angleA = math.degrees(math.acos((b * b + c * c - a * a) / (2 * b * c)))
    return angleA


# ##### 取得兩條線座標之間的角度

# In[ ]:


def get_angle_between_two_lines_position(
    line1: list[list[float, float], list[float, float]],
    line2: list[list[float, float], list[float, float]],
) -> float:
    """取得兩條線座標之間的角度

    Args:
        line1 (list[list[float, float], list[float, float]]): 線條 1的兩點座標
        line2 (list[list[float, float], list[float, float]]): 線條 2的兩點座標

    Returns:
        float: 角度(度數)
    """
    pos1a = np.array(line1[0])
    pos1b = np.array(line1[1])
    pos2a = np.array(line2[0])
    pos2b = np.array(line2[1])

    vec1 = pos1a - pos1b
    vec2 = pos2a - pos2b

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return np.rad2deg(np.arccos(np.dot(vec1, vec2) / (norm1 * norm2)))


# ##### 取得多組點之間的距離

# In[7]:


def get_dist_between_points(*point_list: list[list[float]]) -> float:
    """取得多組點之間的距離

    Args:
        point_list (list[float]): 點列表

    Returns:
        float: 多組點之間的距離
    """
    dist = 0

    if len(point_list) > 1:
        pt1 = point_list[0]
        for pt2 in point_list[1:]:
            start_pt = np.array(pt1) # 起點
            end_pt = np.array(pt2) # 終點
            dist += np.linalg.norm(start_pt - end_pt) # 起點和終點距離
            pt1 = end_pt.tolist()

    return float(dist)


# ##### 取得三角形重心座標

# In[ ]:


def get_triangle_gravity_position(
    pos1: list[float, float, float],
    pos2: list[float, float, float],
    pos3: list[float, float, float],
) -> list[float, float, float]:
    """取得三角形重心座標

    Args:
        pos1 (list[float, float, float]): 座標 1
        pos2 (list[float, float, float]): 座標 2
        pos3 (list[float, float, float]): 座標 3

    Returns:
        list[float, float, float]: 重心座標
    """
    p1 = np.array(pos1)
    p2 = np.array(pos2)
    p3 = np.array(pos3)

    return ((p1 + p2 + p3) / 3).tolist()


# ##### 將角度轉換成 LHC 姿勢標籤

# In[ ]:


def angles_to_lhc_label(angles: dict) -> str:
    """將角度轉換成 LHC 姿勢標籤

    標籤描述:
        A1: 站立
        A2: 搬運高處物品
        A3: 微彎腰
        A4: 彎腰
        A5: 蹲姿、跪姿、跪坐姿勢

    醫師判斷規則:
        A1: 膝蓋 90 - 180 度，腰部 160 - 180 度，肩膀 0 - 90 度
        A2: 膝蓋 90 - 180 度，腰部 160 - 180 度，肩膀 90 - 180 度
        A3: 膝蓋 90 - 180 度，腰部 120 - 160 度
        A4: 膝蓋 90 - 180 度，腰部 0 - 120 度
        A5: 膝蓋 0 - 90 度

    Args:
        angles (dict): 角度字典

    Returns:
        str: LHC 姿勢標籤
    """

    label = ""  # 姿勢標籤

    # 如果無法取得角度
    if len(angles) <= 0:
        label = ""
    ### 左右邊關節只要觸發其中一種條件，就可以直接定義姿勢標籤了
    elif angles["left_knee"] < 90 or angles["right_knee"] < 90:
        label = "A5"
    # 角度區間之後 A4 會設定臀部角度為 0-115 度
    elif angles["left_hip"] < 120 or angles["right_hip"] < 120:
        label = "A4"
    # 角度區間之後 A3 會設定臀部角度為 115-160 度
    elif angles["left_hip"] < 160 or angles["right_hip"] < 160:
        label = "A3"
    elif angles["left_shoulder"] > 90 or angles["right_shoulder"] > 90:
        label = "A2"
    else:
        label = "A1"

    return label


# ##### 將一邊的角度轉換成 LHC 姿勢標籤

# In[ ]:


def angles_to_lhc_label_by_one_side(angles: dict, is_left=False) -> str:
    """將一邊的角度轉換成 LHC 姿勢標籤

    標籤描述:
        A1: 站立
        A2: 搬運高處物品
        A3: 微彎腰
        A4: 彎腰
        A5: 蹲姿、跪姿、跪坐姿勢

    醫師判斷規則:
        A1: 膝蓋 90 - 180 度，腰部 160 - 180 度，肩膀 0 - 90 度
        A2: 膝蓋 90 - 180 度，腰部 160 - 180 度，肩膀 90 - 180 度
        A3: 膝蓋 90 - 180 度，腰部 120 - 160 度
        A4: 膝蓋 90 - 180 度，腰部 0 - 120 度
        A5: 膝蓋 0 - 90 度

    Args:
        angles (dict): 角度字典
        is_left (bool, optional): 是否為左側姿勢. Defaults to False.

    Returns:
        str: LHC 姿勢標籤
    """

    label = ""  # 姿勢標籤

    if is_left:
        # 如果無法取得角度
        if len(angles) <= 0:
            label = ""
        elif angles["left_knee"] < 90:
            label = "A5"
        elif angles["left_hip"] < 120:
            label = "A4"
        # 角度區間之後會設定為 120-150 度，否則站立判斷過於嚴格
        elif angles["left_hip"] < 160:
            label = "A3"
        elif angles["left_shoulder"] > 90:
            label = "A2"
        else:
            label = "A1"
    else:
        # 如果無法取得角度
        if len(angles) <= 0:
            label = ""
        elif angles["right_knee"] < 90:
            label = "A5"
        elif angles["right_hip"] < 120:
            label = "A4"
        # 角度區間之後會設定為 120-160 度，否則站立判斷過於嚴格
        elif angles["right_hip"] < 160:
            label = "A3"
        elif angles["right_shoulder"] > 90:
            label = "A2"
        else:
            label = "A1"

    return label


# ##### 將標籤列表轉換成標籤列表規則

# In[ ]:


def label_list_to_label_list_rule(labels: list[str]) -> list[list[str, int]]:
    """將標籤列表轉換成標籤列表規則
    
    從標籤列表中找到規則

    Args:
        labels (list[str]): 標籤列表

    Returns:
        list[list[str, int]]: 標籤列表的規則
    """

    # 標籤列表的規則
    result = []

    buffer = None  # 暫存標籤資料
    num = 1
    for lab in labels:
        # 初始化參數取得
        if buffer is None:
            buffer = lab
        # 當目前項目等於暫存標籤名稱
        if buffer == lab:
            num += 1
        # 當目前項目不等於暫存標籤名稱
        if buffer != lab:
            result.append([buffer, num])  # 添加資料到結果陣列
            # 重置資料
            buffer = lab
            num = 1

    result.append([buffer, num])  # 添加最後一項資料

    return result


# ##### 將標籤列表規則轉換成標籤列表

# In[ ]:


def label_list_rule_to_label_list(label_list_rule: list[list[str, int]]) -> list[str]:
    """將標籤列表規則轉換成標籤列表

    Args:
        labels (list[list[str, int]]): 標籤列表的規則

    Returns:
        list[str]: 標籤列表
    """

    # 標籤列表
    result = []

    # 添加標籤到標籤列表
    for label, num in label_list_rule:
        result += [label] * int(num)

    return result

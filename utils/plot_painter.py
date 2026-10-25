#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# 格式參考
from matplotlib.pyplot import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# In[ ]:


def set_square_data_range(
    data_range: list[float, float], ax: Axes | Axes3D
) -> plt.Axes:
    """設定圖表資料範圍

    Args:
        data_range (list[float, float]): 資料範圍
        ax (Axes | Axes3D): 座標資料

    Returns:
        plt.Axes: 座標資料
    """
    ax.set_xlim(data_range)
    ax.set_ylim(data_range)
    if ax.name == "3d":
        ax.set_zlim(data_range)

    return ax


# In[ ]:


def set_data_range(
    x: list[float, float],
    y: list[float, float],
    z: list[float, float],
    ax: Axes | Axes3D,
) -> plt.Axes:
    """設定圖表資料範圍

    Args:
        x (list[float, float]): x 軸範圍
        y (list[float, float]): y 軸範圍
        z (list[float, float]): z 軸範圍
        ax (Axes | Axes3D): 座標資料

    Returns:
        plt.Axes: 座標資料
    """
    ax.set_xlim(x)
    ax.set_ylim(y)
    if ax.name == "3d":
        ax.set_zlim(z)

    return ax


# In[2]:


def draw_pose_result_line_chart(
    labels: list[str], data_type: str = None, ax: Axes = None
) -> plt.Axes:
    """取得姿勢結果折線圖

    Args:
        labels (list[str]): 姿勢標籤列表
        data_type (str, optional): 所代表的模型. Defaults to None.
        ax (Axes, optional): 座標資料. Defaults to None.

    Returns:
        plt.Axes: 座標資料
    """

    POSE_LABEL = ["A1", "A2", "A3", "A4", "A5"]

    data = []
    # 將標籤以數值的形式存入到 data 中
    for lab in labels:
        if lab == "A1":
            data.append(0)
        if lab == "A2":
            data.append(1)
        if lab == "A3":
            data.append(2)
        if lab == "A4":
            data.append(3)
        if lab == "A5":
            data.append(4)

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel("time frame")
    ax.set_ylabel("pose label")
    ax.set_yticks(list(range(5)), POSE_LABEL)
    ax.plot(range(len(data)), data, label=data_type)

    return ax


# In[3]:


def draw_dots(
    positions: list[list[float, float]] | list[list[float, float, float]],
    is_3d: bool = False,
    dot_size: int = 5,
    color: str = "#f00",
    label: str = "",
    ax: Axes | Axes3D = None,
) -> plt.Axes:
    """繪製多個點

    Args:
        positions (list[list[list[float, float]]] | list[list[list[float, float, float]]]): 點座標陣列
        is_3d (bool, optional): 是否為 3D 座標. Defaults to False.
        dot_size (int, optional): 點大小. Defaults to 5.
        color (str, optional): 顏色(16位元rgb). Defaults to "#f00".
        ax (plt.Axes, optional): 圖表坐標. Defaults to None.

    Returns:
        plt.Axes: 圖表坐標
    """

    if ax is None:
        ax = plt.gca()

    # 轉換座標點格式
    pos = np.array(positions)

    # 設定標籤
    if not is_3d:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")

    # 繪製多個點
    x, y = pos[:, 0], pos[:, 1]  # 點的 x, y 座標
    if not is_3d:  # 2D 格式
        ax.scatter(x, y, color=color, s=dot_size, label=label)
    else:  # 3D 格式
        z = pos[:, 2]  # z 座標點
        ax.scatter(x, z, y, color=color, s=dot_size, label=label)

    return ax


# In[4]:


def draw_lines(
    positions: list[list[list[float, float]]] | list[list[list[float, float, float]]],
    is_3d: bool = False,
    color: str = "#000",
    label: str = "",
    ax: Axes | Axes3D = None,
) -> Axes | Axes3D:
    """繪製多組線條

    Args:
        positions (list[list[list[float, float]]] | list[list[list[float, float, float]]]): 座標點列表，存放多組的兩個點(用於連線)
        is_3d (bool, optional): 是否為 3D 座標. Defaults to False.
        color (str, optional): 顏色(16位元rgb). Defaults to "#000".
        ax (Axes | Axes3D, optional): 圖表座標. Defaults to None.

    Returns:
        Axes | Axes3D: 圖表座標
    """
    if ax is None:
        ax = plt.gca()

    pos = np.array(positions)  # 取得座標點位置
    colors = [color] * pos.shape[0]  # 設定顏色陣列

    # 設定標籤和設定線條集合
    if not is_3d:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        lines = LineCollection(pos, colors=colors, label=label)
        ax.add_collection(lines)
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")

        # 取得三維座標點
        x = pos[:, :, 0]
        y = pos[:, :, 1]
        z = pos[:, :, 2]

        # 將舊的三維座標點映射在新的物件上
        new_pos = pos.copy()
        new_pos[:, :, 0] = x
        new_pos[:, :, 1] = z
        new_pos[:, :, 2] = y

        lines = Line3DCollection(new_pos, colors=colors, label=label)
        ax.add_collection3d(lines)

    return ax


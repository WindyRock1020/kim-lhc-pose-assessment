#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


def scaleYAxis(data: list, val: float) -> list:
    """縮放 Y 軸座標(只作用於二三維陣列)

    Args:
        data (list): 被縮放的目標
        val (float): 縮放數值

    Returns:
        list: 縮放結果
    """

    data_arr = np.array(data)

    layer_size = len(data_arr.shape)
    if layer_size == 2:  # 二維陣列
        data_arr[:, 1] *= val
    elif layer_size == 3:  # 三維陣列
        data_arr[:, :, 1] *= val

    return data_arr.tolist()


# In[ ]:


def transYAxis(data: list, val: float) -> list:
    """平移 Y 軸座標(只作用於二三維陣列)

    Args:
        data (list): 被平移的目標
        val (float): 平移數值

    Returns:
        list: 平移結果
    """
    data_arr = np.array(data)

    layer_size = len(data_arr.shape)
    if layer_size == 2:  # 二維陣列
        data_arr[:, 1] += val
    elif layer_size == 3:  # 三維陣列
        data_arr[:, :, 1] += val

    return data_arr.tolist()


# In[ ]:


def translate_a_kpt_to_plot_pos(all_kpt_pos: list, origin_kpt_pos: list) -> list:
    """轉換單個關鍵點座標轉換成圖表座標

    Args:
        all_kpt_pos (list): 所有關鍵點座標(用於設定 y 軸平移量)
        origin_kpt_pos (list): 原始關鍵點座標

    Returns:
        list: 圖表座標
    """
    all_kpts = np.array(all_kpt_pos)
    add_y = all_kpts[:, 1].max()

    return transYAxis(scaleYAxis(origin_kpt_pos, -1), add_y)


# In[ ]:


def translate_multi_kpt_to_plot_pos(
    all_kpt_pos: list, *origin_kpt_args: list[list]
) -> list:
    """將多個關鍵點座標轉換成圖表座標

    Args:
        all_kpt_pos (list): 所有關鍵點座標(用於設定 y 軸平移量)
        origin_kpt_args (list[list]): 多組原始關鍵點座標

    Returns:
        list: 圖表座標
    """
    kpt_pos_list = []
    for origin_kpt_pos in origin_kpt_args:
        kpt_pos_list.append(translate_a_kpt_to_plot_pos(all_kpt_pos, origin_kpt_pos))

    return kpt_pos_list


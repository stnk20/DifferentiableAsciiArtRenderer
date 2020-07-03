"""
AA生成に必要なデータを準備する
"""
import os
import numpy as np
import pickle as pkl
import pandas as pd

from param import TEMPLATE_NAME, MASK_NAME, CHAR_HEIGHT, CHAR_WIDTH, CHAR_DIM

CHAR_DICT_NAME = "./DeepAA/data/char_dict.pkl"
CHAR_LIST_NAME = "./DeepAA/data/char_list.csv"


if __name__ == "__main__":
    # char_to_template
    with open(CHAR_DICT_NAME, "rb") as f:
        template_dict = pkl.load(f)

    df = pd.read_csv(CHAR_LIST_NAME, encoding="cp932")
    # index_to_char
    char_list = df.sort_index()["char"].values.tolist()[:CHAR_DIM]  # 元ファイルで出現回数順に並んでいるので上からCHAR_DIMだけ利用

    # prepare template.npy and mask.npy
    if not os.path.exists(TEMPLATE_NAME) or not os.path.exists(MASK_NAME):
        template = np.zeros((CHAR_HEIGHT, CHAR_WIDTH, 1, CHAR_DIM))
        mask = np.zeros((CHAR_HEIGHT, CHAR_WIDTH, 1, CHAR_DIM))
        for i, c in enumerate(char_list):
            char_img = 1.0*template_dict[c]
            h, w = char_img.shape
            template[:h, :w, 0, i] = char_img
            mask[:, :w, 0, i] = 1
        np.save(TEMPLATE_NAME, template.astype(np.float32))
        np.save(MASK_NAME, mask.astype(np.float32))

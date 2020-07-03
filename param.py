TEMPLATE_NAME = "./template.npy"
MASK_NAME = "./mask.npy"
DRAW_GLOB = "/home/satoshi/dataset/photo_sketching/sketch-rendered/width-1/*.png"
# DRAW_GLOB = "/home/satoshi/dataset/multicue/ground-truth/images/edges/*.png"
# 文字の情報
CHAR_HEIGHT = 18
CHAR_WIDTH = 16  # 文字の最大幅
CHAR_DIM = 324  # 最大899　変更後はtemplate.npyとmask.npyを削除してprepare.pyを実行すること
# 学習用の画像サイズ
IMG_HEIGHT = 18*8
IMG_WIDTH = 18*8

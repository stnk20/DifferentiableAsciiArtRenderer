from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

from param import *
from layers import *


def build_encoder(img_height, img_width):
    shift = 16  # 右方向の情報で位置調整などを行うことになるため、受容野を右に寄せる
    x = Input(shape=(img_height, img_width, 1))  # range: 0 to 1
    xp = ZeroPadding2D(((0, 0), (shift, 0)))(x)

    h = Conv2D(32, (5, 5), strides=(2, 1), padding="same")(xp)
    h = Activation("relu")(h)
    h = Conv2D(32, (3, 3), dilation_rate=(1, 2), padding="same")(h)
    h = BatchNormalization()(h)

    h = Activation("relu")(h)
    h = Conv2D(64, (3, 3), dilation_rate=(1, 2), padding="same")(h)
    h = h[:, 1::3, :, :]  # diration_rateとstrideを同時に指定できないためここでstride反映
    h = BatchNormalization()(h)

    r = h
    h = Activation("relu")(h)
    h = Conv2D(64, (3, 3), dilation_rate=(1, 6), padding="same")(h)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Conv2D(64, (3, 3), dilation_rate=(1, 6), padding="same")(h)
    h = BatchNormalization()(h)
    h = Add()([r, h])

    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Conv2D(128, (3, 3), dilation_rate=(1, 6), padding="same")(h)
    h = h[:, 1::3, :, :]  # diration_rateとstrideを同時に指定できないためここでstride反映
    h = BatchNormalization()(h)

    r = h
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Conv2D(128, (3, 3), dilation_rate=(1, 18), padding="same")(h)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Conv2D(128, (3, 3), dilation_rate=(1, 18), padding="same")(h)
    h = BatchNormalization()(h)
    h = Add()([r, h])

    h = BatchNormalization()(h)
    h = Activation("relu")(h)
    h = Conv2D(512, (1, 1), padding="same")(h)
    h = BatchNormalization()(h)
    h = Activation("relu")(h)

    h = Dropout(0.5)(h)

    h = h[:, :, :-shift, :]

    # 文字候補
    # スペースが多くなるのでlogitの値域が大きくなりすぎないようにsoftmaxを2段にする
    # 1段目：全角,半角,文字あり　2段目：文字の種類
    # dropoutを効果的にする意図でbiasを0に固定
    yc_logit_blank = Conv2D(3, (1, 1), padding="same", use_bias=False)(h)
    yc_logit_char = Conv2D(CHAR_DIM-2, (1, 1), padding="same", use_bias=False)(h)

    yc_blank = K.softmax(yc_logit_blank)
    yc_char = K.softmax(yc_logit_char)

    yc = K.concatenate([yc_blank[:, :, :, 0:2], yc_blank[:, :, :, 2:3]*yc_char])

    return Model(inputs=[x], outputs=[yc])


def circle_dilate(x, r0, r1):
    # r0まで0、r1で-1になるようなカーネルを作ってdilate実行
    ri = int(np.floor(r1))
    ksize = ri*2+1
    t = np.linspace(-ri, ri, ksize)
    r = (np.zeros((ksize, ksize))+t[:, np.newaxis]**2+t[np.newaxis, :]**2)**0.5
    k = np.minimum(0.0, (r-r0)/(r0-r1))[:, :, np.newaxis]
    return tf.nn.dilation2d(x, k, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1))


def calc_feature(image, feature_model):
    # 半分にリサイズ
    image = K.pool2d(image, (2, 2), (2, 2), pool_mode="avg")
    # コントラスト・符号を調整
    a = -1  # 黒線
    image = a*image
    # 背景が0の線画であるためpreprocess_inputはスキップ
    return feature_model(K.concatenate([image]*3))


if __name__ == "__main__":
    model = build_encoder(100, 100)
    model.summary()

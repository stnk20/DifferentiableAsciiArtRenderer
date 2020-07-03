import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import tqdm
import os
import glob
import random
import numpy as np
import cv2
import fusewarp

from param import *
from models import *


def data_generator(batch_size, width=IMG_WIDTH, height=IMG_HEIGHT):
    threshold = 0.3
    warp = fusewarp.FuseWarp([
        fusewarp.transform.Scale((1.0, 3.0)),
        fusewarp.transform.FrameRandom(width, height),
        fusewarp.transform.FlipLR(),
        fusewarp.transform.Rotate((-50, 50)),
        ])
    while True:
        img_names = glob.glob(DRAW_GLOB)
        Y = np.zeros((batch_size, height, width, 1), dtype=np.float32)  # 画像
        for i in range(batch_size):
            img = cv2.imread(random.choice(img_names), cv2.IMREAD_GRAYSCALE)
            while True:
                h, w = img.shape[:2]
                warp.next_sample((w, h))
                matrix = warp.get_matrix()
                y0 = cv2.warpPerspective(img, matrix, (width, height), borderValue=255)
                if np.mean(y0) < 255:
                    break
            _, yt = cv2.threshold(y0, 255*(1-threshold), 255, cv2.THRESH_BINARY_INV)  # 背景が黒になる
            yt = cv2.ximgproc.thinning(yt, thinningType=cv2.ximgproc.THINNING_GUOHALL)  # THINNING_ZHANGSUEN / THINNING_GUOHALL 後者のほうが斜線が細くなる
            Y[i, :, :, 0] = yt/255
        yield Y


def main(
        initial_lr=0.001,
        batch_size=16,
        n_steps=100,
        n_epochs=200,
        initial_model_path=""):

    if not os.path.exists("model"):
        os.mkdir("model")

    datagen = data_generator(batch_size)

    encoder_model = build_encoder(IMG_HEIGHT, IMG_WIDTH)
    encoder_optimizer = RMSprop(clipvalue=1.0)

    if initial_model_path != "":
        encoder_model.load_weights(initial_model_path)

    vgg = keras.applications.vgg19.VGG19(include_top=False, input_shape=(IMG_HEIGHT//2, IMG_WIDTH//2, 3))
    feature_model = Model(inputs=vgg.inputs, outputs=[vgg.get_layer("block3_conv1").output])
    feature_model.trainable = False

    template = K.constant(np.load(TEMPLATE_NAME))
    mask_template = K.constant(np.load(MASK_NAME))

    @tf.function
    def train_step(y_true, tau_st):
        reg_coeff = 5
        r0, r1 = 0.5, 2.5
        decay_placement_grad = 0.5
        n = 16  # gumbel_softmaxのサンプリングの影響を軽減するために複数回実行する
        with tf.GradientTape() as tape:
            candidate_prob = encoder_model(y_true, training=True)

            f_true = calc_feature(y_true, feature_model)
            d_true = circle_dilate(y_true, r0, r1)

            y_true = K.concatenate([y_true]*n, axis=0)
            f_true = K.concatenate([f_true]*n, axis=0)
            d_true = K.concatenate([d_true]*n, axis=0)

            candidate_prob = K.concatenate([candidate_prob]*n, axis=0)
            candidate = gumbel_softmax(candidate_prob, tau=tau_st, straight_through=True)
            placement = decide_placement(candidate, mask_template, decay_grad=decay_placement_grad)
            y_pred = draw_image(candidate, placement, template)

            f_pred = calc_feature(y_pred, feature_model)

            main_loss = K.mean((f_true-f_pred)**2)

            # jaccard-like
            d_pred = circle_dilate(y_pred, r0, r1)
            reg_loss = reg_coeff*(1.0-K.mean(K.minimum(d_true, d_pred))/K.mean(K.maximum(d_true, d_pred)))

            loss = main_loss+reg_loss

        gradients = tape.gradient(loss, encoder_model.trainable_variables)
        encoder_optimizer.apply_gradients(zip(gradients, encoder_model.trainable_variables))

        return loss, reg_loss

    with open("log.csv", "w") as f:
        f.write("epoch, loss, reg_loss\n")
        for epoch in range(1, n_epochs+1):
            loss = reg_loss = 0
            tau_st = tau_st = max(0.2, 2*0.1**(epoch/100))
            encoder_optimizer.lr = initial_lr*min(1.0, 0.1**((epoch-100)/100))
            with tqdm.tqdm(range(n_steps)) as pbar:
                pbar.set_description(f"epoch {epoch}")
                for i in pbar:
                    y = next(datagen)
                    step_losses = train_step(y, tf.constant(tau_st))  # 関数の再トレースを防ぐためにtf.constantを使う ref: https://www.tensorflow.org/tutorials/customization/performance?hl=ja#%E5%BC%95%E6%95%B0%E3%81%AF_python_%E3%81%8B%EF%BC%9F_tensor_%E3%81%8B%EF%BC%9F
                    loss += step_losses[0].numpy()
                    reg_loss += step_losses[1].numpy()
                    pbar.set_postfix(loss=f"{loss/(i+1):.5g}", reg_loss=f"{reg_loss/(i+1):.5g}")
            f.write(f"{epoch}, {loss/(i+1)}, {reg_loss/(i+1)}\n")
            f.flush()
            encoder_model.save_weights(f"model/weights{epoch:04d}.h5")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

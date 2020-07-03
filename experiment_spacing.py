import tensorflow as tf
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import tqdm
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from param import TEMPLATE_NAME, MASK_NAME, CHAR_HEIGHT
from layers import hardmax, gumbel_softmax, decide_placement, draw_image

def main():
    # 全角・半角スペース・ピリオドの組み合わせを適切に選んで位置調整を学習する実験

    batch_size = 64
    width = 40
    rows = width-9
    decay_placement_grad = 0.8  # ここが0だと逆伝播が上手く行かない
    tau = 1

    # 文字を4種類に限定（全角スペース・半角スペース・縦棒・ピリオド）
    template = K.constant(np.load(TEMPLATE_NAME)[:, :, :, [0, 1, 2, 10]])
    mask_template = K.constant(np.load(MASK_NAME)[:, :, :, [0, 1, 2, 10]])

    # 文字候補のlogitを直接最適化する
    candidate_logit = K.variable(np.zeros((1, rows, width, 4)), dtype="float32")
    optimizer = Adam(0.01)

    # データ生成
    y_true = np.zeros((batch_size, CHAR_HEIGHT*rows, width, 1), dtype=np.float32)
    for i in range(rows):
        # 全角スペースが11、半角スペースが5、ピリオドが3なので、8以上であれば可能な組み合わせがある（縦棒は010の形なので+1だけオフセットする）
        y_true[:, CHAR_HEIGHT*i:CHAR_HEIGHT*i+16, i+9, :] = 1

    @tf.function
    def train_step(y_true):
        with tf.GradientTape() as tape:
            candidate_prob = K.softmax(candidate_logit)
            candidate_prob = K.concatenate([candidate_prob]*batch_size, axis=0)
            candidate = gumbel_softmax(candidate_prob, tau=tau, straight_through=True)  # STなしのほうが学習が安定している印象
            placement = decide_placement(candidate, mask_template, decay_grad=decay_placement_grad)
            y_pred = draw_image(candidate, placement, template)

            loss = K.mean((y_true-y_pred)**2)

        gradients = tape.gradient(loss, candidate_logit)
        optimizer.apply_gradients([[gradients, candidate_logit]])

        return loss, candidate_prob, y_pred

    with tqdm.tqdm(range(1000)) as pbar:
        for i in pbar:
            loss, candidate_prob, y_pred = train_step(y_true)

            # deterministic
            candidate_d = hardmax(candidate_prob)
            placement_d = decide_placement(candidate_d, mask_template, decay_grad=0)
            y_d = draw_image(candidate_d, placement_d, template)

            loss_d = K.mean((y_true-y_d)**2)
            pbar.set_postfix(loss=f"{loss:.3g}", loss_deterministic=f"{loss_d:.3g}")

            cv2.imshow("true", y_true[0, :, :, 0])
            cv2.imshow("prob", K.mean(y_pred[:, :, :, 0], axis=0).numpy())
            cv2.imshow("deterministic", y_d[0, :, :, 0].numpy())
            # cv2.imshow("candidate", np.stack([candidate_prob[0, 0, :, :].numpy()]*CHAR_HEIGHT, axis=0))
            k = cv2.waitKey(1)
            if k == 27:
                return
    cv2.waitKey(0)


if __name__ == "__main__":
    import fire
    fire.Fire(main)

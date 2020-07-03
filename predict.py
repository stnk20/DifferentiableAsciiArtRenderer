from train import data_generator
from models import *
from param import *
import cv2
import numpy as np
import tensorflow.keras.backend as K
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def array2texts(candidate, mask_template):
    char_list = ['\u3000', ' ', '|', '/', '_', '＼', '／', '＿', '￣', ',', '.', 'l', '-', "'", '{', '}', '‐', '〉', 'ヽ', '´', '〈', '｀', 'ｌ', '､', '〔', '二', '=', '〕', 'r', '―', 'ﾉ', 'i', '′', '⌒', '∧', '、', 'ノ', '∨', '｜', ':', '`', 'ｰ', 'j', '¨', 'ﾆ', 'ｉ', '｛', '三', '丶', '!', '｢', 'ﾊ', 'ー', '｝', '＞', '人', '[', '八', ']', '＜', 'く', 'ﾍ', 'ﾄ', 'Y', ')', 'ミ', '(', 'ニ', '彡', 'イ', 'V', '∠', '｣', '7', '（', 'ｲ', '└', 'ト', '＾', '"', '…', '）', 'ハ', '─', '┐', '┘', '．', '厶', '^', 'Ｖ', 'ｧ', ';', '辷', 'x', 'ﾞ', '〃', '「', '￤', '＝', '７', '>', 'ﾐ', '┴', 'ﾘ', '厂', 'ｨ', 'ﾟ', 'ﾌ', '━', '‘', 'L', '从', 'ｘ', 'T', '乂', '」', 'f', '≧', '！', 'ゝ', 'Ｙ', 'v', '≦', '丿', '匚', '｡', 'ﾏ', 'ィ', 'X', '┌', '￢', '：', '冖', '┬', '斗', 'ア', '，', '│', '~', 'ヾ', '个', '小', 'Ⅵ', 'ん', '⊥', '廴', '≠', 'う', 'ｱ', 'し', '弋', 'o', 'ﾚ', 'ソ', 'ﾋ', '’', 'て', '仁', '爪', 'Ｌ', 'ｭ', 'ｿ', '･', 'k', '丁', '‥', 'い', '一', '。', '工', '》', 'ヘ', 'y', 'z', '◯', '宀', '刈', 'ｆ', '～', '<', 't', 'ゞ', '］', 'ｎ', '⊂', 'O', 'ｒ', '了', 'ﾒ', '⊃', '☆', '匸', 'Ｎ', '┼', '・', 'ﾑ', 'Λ', '《', '゜', 'ァ', 'こ', '1', 'Ｔ', 'n', '≫', '┤', 'リ', 'Z', 'ⅰ', 'り', '勹', 'メ', '寸', 'H', 'フ', 'レ', '代', 'ﾃ', 'ﾝ', 'へ', 'ｊ', 'ぅ', '癶', 'ｈ', 'ツ', '下', '介', '；', 'ｚ', 'N', '″', '├', '１', 'ｪ', 'u', 'マ', '冫', 'ｬ', '孑', 'ｋ', '十', '乙', '３', 'ﾖ', 'ﾛ', '≡', '㌻', '入', '＋', '゛', '心', 'ｽ', '°', '仏', '示', '込', '+', 'J', 'c', '┃', '广', 'I', '┓', '▽', '0', '√', '≪', 'っ', '不', 'Ｏ', 'ﾂ', '┛', '七', '［', 'ｼ', 'ﾅ', 'У', '〇', 'つ', 'ン', '抖', 'ｯ', 'ム', '千', '辻', 'ｔ', 'ｕ', '云', '弌', 'ⅱ', '∟', 'ヒ', '少', '才', '芋', 'ｖ', 'ﾕ', '┏', '┰', '㍉', '以', '儿', '升', '斤', '－', 'ｷ', 'h', 'Ⅳ', '【', 'た', 'づ', 'ス', '圦', '川', '汽', 'ｸ', 'ﾁ', '?', 'w', 'エ', 'ロ', '卜', '彳', '%', 'と', '㏍', '上', '匕', '笊', 'Ｃ', 'Ｘ', 'ﾙ', '○', 'キ', '弐', 'Ａ', 'ｏ', 'W', '×', '炒', '筏', 'C', 'P', '∵', '≒', 'か', 'タ', 'ユ', 'ル', '乃', '冂', '戈', '扞', '气', '汀', 'Ｗ', 'ｴ', 'A', '∩', '】', '狄', '行', '”', '┗', '┣', '●', 'じ', 'ナ', '匀', '大', '苧', '＊', 'K', 'U', 'シ', '仄', '土', '弖', 'Ｚ', 'ｾ', '‰', 'ⅵ', '⊇', 'で', '于', '仆', '干', '爻', '襾', 'Ｕ', 'ﾀ', '℃', 'ぃ', 'え', 'ク', 'ド', '壬', '沁', '荻', 'ｺ', 'S', 'Τ', '“', 'テ', 'ュ', '乞', '庁', '％', 'Ｈ', 'Ｊ', 'ｃ', 'に', 'チ', '且', '丨', '仕', '弓', '忙', '灯', '爿', 'ｦ', 'ﾓ', 'Ξ', 'Ⅹ', '⊆', 'ヤ', '卞', '士', '沙', '＂', 'Ｋ', 'E', 'F', 'G', 'a', 'Ν', '┫', '△', 'だ', 'コ', 'ヨ', 'ヮ', '乢', '亅', '仟', '仡', '几', '打', '芥', 'ｩ', 'ﾇ', '*', '¶', 'Χ', '┯', '〆', '〟', 'の', 'ひ', 'れ', 'カ', 'ヱ', '丕', '主', '似', '圷', '弄', '托', '抃', '竺', '芍', '非', '＇', 'ｗ', 'Υ', '∪', '♪', '『', '〒', 'ぐ', '㌧', '㎡', '仍', '兆', '刋', '及', '口', '夲', '幵', '廾', '必', '斥', '洲', '片', '王', '瓜', '禾', '０', 'ｙ', 'ｶ']
    placement = decide_placement(candidate, mask_template, decay_grad=0)
    batch_size, height, width = candidate.shape[:3]
    ret = []
    for b in range(batch_size):
        text = ""
        for i in range(height):
            for j in range(width):
                if placement[b, i, j, 0] > 0.999:
                    c = np.argmax(candidate[b, i, j])
                    text += char_list[c]
            text += "\r\n"
        ret.append(text)
    return ret


def main(model_path, img_path="", tau=0, deterministic=True, save_image=False):
    if img_path != "":
        def g():
            img_path_list = sorted(glob.glob(img_path))
            i = 0
            while True:
                y_true = cv2.imread(img_path_list[i % len(img_path_list)], cv2.IMREAD_GRAYSCALE)
                _, y_true = cv2.threshold(y_true, 5, 1, cv2.THRESH_BINARY_INV)
                height, width = y_true.shape[:2]
                height = (height//CHAR_HEIGHT)*CHAR_HEIGHT
                y_true = y_true[:height, :].reshape((1, height, width, 1)).astype(np.float32)
                i += 1
                yield y_true
        gen = g()
    else:
        gen = data_generator(10, IMG_WIDTH*2, IMG_HEIGHT*2)

    encoder = build_encoder(IMG_HEIGHT, IMG_WIDTH)
    if model_path != "":
        encoder.load_weights(model_path)

    template = K.constant(np.load(TEMPLATE_NAME))
    mask_template = K.constant(np.load(MASK_NAME))

    write_idx = 0
    while True:
        y_true = next(gen)
        encoder.input.batch_input_shape = y_true.shape

        candidate_prob = encoder(y_true, training=False)
        if deterministic:
            candidate = hardmax(candidate_prob)
        else:
            candidate = gumbel_softmax(candidate_prob, tau=tau)
        placement = decide_placement(candidate, mask_template, decay_grad=0)
        y_pred = draw_image(candidate, placement, template)
        # m_pred = draw_mask(candidate, placement, mask_template, space_only=True)

        for i in range(y_true.shape[0]):
            cv2.imshow("true", 1-y_true[i])
            cv2.imshow("pred", 1-y_pred[i].numpy())
            cv2.imshow("", np.concatenate([y_true[i], np.zeros_like(y_true[i]), y_pred[i].numpy()], axis=-1))
            # cv2.imshow("mask", m_pred[i].numpy()/2)

            if save_image:
                write_idx += 1
                cv2.imwrite(f"{write_idx:02d}.png", (255-255*y_pred[i].numpy()).astype(np.uint8))

            k = cv2.waitKey()
            if k == 27:
                exit(0)
            if k == ord("w"):  # press w key to output AA
                print(array2texts(candidate[i:i+1], mask_template)[0])
            if k == ord("d"):
                deterministic = not deterministic
                print("deterministic:", deterministic, "(valid from next batch)")


if __name__ == "__main__":
    import fire
    fire.Fire(main)

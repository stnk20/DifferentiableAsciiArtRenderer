import tensorflow as tf
from tensorflow.keras import backend as K


def combine_value_gradient(xv, xg):
    # 値とgradientで別のフローを適用する
    return tf.stop_gradient(xv)+tf.custom_gradient(lambda x: [K.zeros_like(x), lambda dy: dy])(xg)


def step(x):
    return K.cast_to_floatx(K.greater(x, 0.0))


def gumbel_sigmoid(x, tau, from_logits=False, straight_through=False):
    # ref: https://arxiv.org/abs/1611.01144
    # ref: https://arxiv.org/abs/1611.00712
    eps = 1e-20
    u = K.random_uniform(K.shape(x), eps, 1-eps)
    if not from_logits:
        x = K.log(K.maximum(eps, x))-K.log(K.maximum(eps, 1-x))  # prob->logit
    y = x+K.log(u)-K.log(1-u)
    if tau > 0:
        if straight_through:
            return combine_value_gradient(step(y), K.sigmoid(y/tau))
        else:
            return K.sigmoid(y/tau)
    else:
        return step(y)


def hardmax(x):
    return K.one_hot(K.argmax(x, axis=-1), K.int_shape(x)[-1]) 


def gumbel_softmax(x, tau, from_logits=False, straight_through=False):
    # ref: https://arxiv.org/abs/1611.01144
    eps = 1e-20
    u = K.random_uniform(K.shape(x), eps, 1-eps)
    if not from_logits:
        x = K.log(K.maximum(eps, x))
    y = x-K.log(-K.log(u))
    if tau > 0:
        if straight_through:
            return combine_value_gradient(hardmax(y), K.softmax(y/tau, axis=-1))
        else:
            return K.softmax(y/tau, axis=-1)
    else:
        return hardmax(y)


def decide_placement(candidate, mask_template, decay_grad):
    # 文字候補をもとに可能な配置を決定する
    _, char_width = mask_template.shape[:2]
    mask_template_T = K.transpose(mask_template[0, :, 0, :])  # shape=(char_dim, char_width)

    def step_fn(inputs, states):  # shape=(batch_size, height, char_dim), (batch_size, height, char_width)
        s = states[0]
        placement_t = combine_value_gradient(1.0-s[:, :, :1], decay_grad*(1.0-s[:, :, :1]))  # 勾配が減衰しないと学習が難しすぎるため対策
        s = s+placement_t*K.dot(inputs, mask_template_T)
        new_state = K.concatenate([s[:, :, 1:], K.zeros_like(s[:, :, :1])])  # shape=(batch_size, height, char_width)
        return placement_t, [new_state]

    initial_state = K.zeros_like(candidate[:, :, :char_width, 0])
    candidate_t = tf.transpose(candidate, perm=[0, 2, 1, 3])
    _, placement_t, _ = K.rnn(step_fn, candidate_t, [initial_state])
    return tf.transpose(placement_t, perm=[0, 2, 1, 3])


def draw_image(candidate, placement, template):
    char_height, char_width = template.shape[:2]
    batch_size, height, width = K.int_shape(candidate)[:3]
    image = K.conv2d_transpose(
        candidate*placement,
        template,
        output_shape=(batch_size, height*char_height, width+char_width-1, 1),
        strides=(char_height, 1),
        padding="valid")
    return image[:, :, :-char_width+1, :]


def draw_mask(candidate, placement, mask_template, space_only=False):
    char_height, char_width = mask_template.shape[:2]
    batch_size, height, width = K.int_shape(candidate)[:3]
    if space_only:
        mask_template = mask_template[:, :, :, :2]
        candidate = candidate[:, :, :, :2]
    mask = K.conv2d_transpose(
        candidate*placement,
        mask_template,
        output_shape=(batch_size, height*char_height, width+char_width-1, 1),
        strides=(char_height, 1),
        padding="valid")
    return mask[:, :, :-char_width+1, :]

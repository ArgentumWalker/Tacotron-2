import functools
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauMonotonicAttention
import tensorflow as tf
from tensorflow.python.ops import math_ops, random_ops, array_ops

'''
Implementation for https://arxiv.org/abs/1906.00672
Tips: The code could be directly used in place of BadahnauMonotonicAttention in Tensorflow codes. Similar to its 
base class in the Tensorflow seq2seq codebase,  you may use "hard" for hard inference, or "parallel" for training or 
soft inference. "recurrent" mode in BadahnauMonotonicAttention is not supported. 
If you have already trained another model using BadahnauMonotonicAttention, the model could be reused, otherwise you 
possibly have to tune the score_bias_init, which, similar to that in Raffel et al., 2017, is determined a priori to 
suit the moving speed of the alignments, i.e. speed of speech of your training corpus in TTS cases. So 
score_bias_init=3.5, is a good one for our data, but not necessarily for yours, and our experiments find that the 
results are sensitive to this bias: When the parameter is deviated from the best value, by, say, a small amount of 
0.5, the whole training process may fail. sigmoid_noise=2.0 is enough in our experiments, but if you found that the 
resultant alignments are far from binary, adding more noise (or annealing the noise) might be useful. Other 
hyperparameters in our experiments simply follow the original Tacotron2 settings, and they work. 
'''


def monotonic_stepwise_attention(p_choose_i, previous_attention, mode):
    # p_choose_i, previous_alignments, previous_score: [batch_size, memory_size]
    # p_choose_i: probability to keep attended to the last attended entry i
    if mode == "parallel":
        pad = tf.zeros_like(previous_attention[:, :1])
        attention = previous_attention * p_choose_i + tf.concat(
            [pad, previous_attention[:, :-1] * (1.0 - p_choose_i[:, :-1])], axis=1)
    elif mode == "hard":
        # Given that previous_alignments is one_hot
        move_next_mask = tf.concat([tf.zeros_like(previous_attention[:, :1]), previous_attention[:, :-1]], axis=1)
        stay_prob = tf.reduce_sum(p_choose_i * previous_attention, axis=1) # [B]
        attention = tf.where(stay_prob > 0.5, previous_attention, move_next_mask)
    else:
        raise ValueError("mode must be 'parallel', or 'hard'.")
    return attention


def _stepwise_monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode, seed=None):
    if sigmoid_noise > 0:
        noise = random_ops.random_normal(array_ops.shape(score), dtype=score.dtype,
                                         seed=seed)
        score += sigmoid_noise * noise
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = math_ops.cast(score > 0, score.dtype)
    else:
        p_choose_i = math_ops.sigmoid(score)
    alignments = monotonic_stepwise_attention(p_choose_i, previous_alignments, mode)
    return alignments


class BahdanauStepwiseMonotonicAttention(BahdanauMonotonicAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=True,
                 score_mask_value=None,
                 sigmoid_noise=2.0,
                 sigmoid_noise_seed=None,
                 score_bias_init=3.5,
                 mode="parallel",
                 dtype=None,
                 name="BahdanauStepwiseMonotonicAttention"):
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _stepwise_monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
            seed=sigmoid_noise_seed)
        super(BahdanauMonotonicAttention, self).__init__(
            query_layer=tf.layers.Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=tf.layers.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_bias_init = score_bias_init

    def __call__(self, query, state, prev_max_attentions):
        attention, next_state = super().__call__(query, state)
        return attention, next_state, prev_max_attentions

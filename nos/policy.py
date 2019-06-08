import tensorflow as tf

from stable_baselines.common.policies import *
from .distributions import MaskedCategoricalPDType

class CustomLSTMPolicy(RecurrentActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=128, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=True, feature_extraction="mlp",
                 **kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))

        with tf.variable_scope("input", reuse=False):
            self.action_mask = tf.placeholder(tf.bool, (n_batch, ac_space.n), name="action_mask")

        self._pdtype = MaskedCategoricalPDType(ac_space.n, self.action_mask)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            layers = [64, 64]
        else:
            warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                extracted_features = cnn_extractor(self.processed_obs, **kwargs)
            else:
                extracted_features = tf.layers.flatten(self.processed_obs)
                for i, layer_size in enumerate(layers):
                    extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                        init_scale=np.sqrt(2)))
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                            layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self._value_fn = value_fn
        
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False, action_mask=None):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask, self.action_mask: action_mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask, self.action_mask: action_mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
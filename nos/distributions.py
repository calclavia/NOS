import numpy as np
import tensorflow as tf
from stable_baselines.common import distributions
from stable_baselines.a2c.utils import linear

class MaskedCategoricalPDType(distributions.CategoricalProbabilityDistributionType):
    def __init__(self, n_cat, mask):
        """
        The probability distribution type for categorical input
        :param n_cat: (int) the number of categories
        """
        super().__init__(n_cat)
        self.mask = mask

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', self.n_cat, init_scale=init_scale, init_bias=init_bias)
        
        # These are the logits
        # mask contains all legal actions that can be taken
        # pdparam = tf.where(self.mask, pdparam, tf.broadcast_to(tf.constant([-np.inf]), self.mask.shape))
        pdparam = tf.where(self.mask, pdparam, tf.constant(-1e10, shape=self.mask.shape))
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values
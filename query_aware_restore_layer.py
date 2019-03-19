from __future__ import division
import caffe
import numpy as np

class QueryAwareRestoreLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):

        if self.phase == 0:  # train phase
            import train_config
            config = train_config.Config()
        else:   # val or test phase
            import test_config
            config = test_config.Config()

        self.N = config.N
        self.context_dim = config.context_dim
        self.spatial_dim = config.spatial_dim
        self.HW = config.spatial_pool_map * config.spatial_pool_map
        self.T = config.T
        self.key_word_thresh = config.key_word_thresh
        self.hard_word_att_idx = []

        # query_aware_context features for every image location
        top[0].reshape(self.N, self.context_dim + self.spatial_dim, self.HW)


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):
        # language attention for every image location
        language_att = bottom[0].data  # N*HW*T
        # features for every image location
        img_context = bottom[1].data  # N*C*T
        spatial = bottom[2].data
        spatial_context = bottom[3].data  # N*8*T

        query_aware_context = np.zeros((self.N, self.context_dim + self.spatial_dim, self.HW))
        for img in range(self.N):
            language_att_cur = np.squeeze(language_att[img*self.HW: (img+1)*self.HW, :, :])  # language_att of current image
            img_context_cur = np.squeeze(img_context[img, :, :])  # feature of current image
            spatial_cur = spatial[img, :, :, :]
            spatial_cur = np.reshape(spatial_cur, (self.spatial_dim, self.HW))
            spatial_context_cur = np.squeeze(spatial_context[img, :, :])  # feature of current image

            context_appear_cur = np.zeros((self.context_dim, self.HW))
            context_spatial_cur = np.zeros((self.spatial_dim, self.HW))
            for word in range(self.T):
                word_att = language_att_cur[:, word]
                hard_word_att_idx = np.squeeze(np.where(word_att >= self.key_word_thresh))
                if hard_word_att_idx.size > 0:
                    self.hard_word_att_idx.append(['img, word', img, word])
                    self.hard_word_att_idx.append(hard_word_att_idx)

                    if hard_word_att_idx.size == 1:
                        context_appear_cur[:, hard_word_att_idx] = context_appear_cur[:, hard_word_att_idx] \
                                                                   + img_context_cur[:, word]

                        context_spatial_cur[:, hard_word_att_idx] = context_spatial_cur[:, hard_word_att_idx] \
                                                                    + spatial_cur[:, hard_word_att_idx] \
                                                                    - spatial_context_cur[:, word]
                    else:
                        context_appear_cur[:, hard_word_att_idx] = context_appear_cur[:, hard_word_att_idx] \
                                                               + img_context_cur[:, word][:, np.newaxis]

                        context_spatial_cur[:, hard_word_att_idx] = context_spatial_cur[:, hard_word_att_idx] \
                                                                + spatial_cur[:, hard_word_att_idx] \
                                                                - spatial_context_cur[:, word][:, np.newaxis]
            query_aware_context[img, 0:self.context_dim, :] = context_appear_cur
            query_aware_context[img, self.context_dim:, :] = context_spatial_cur

        top[0].data[...] = query_aware_context

    def backward(self, top, propagate_down, bottom):
        query_aware_context_diff = top[0].diff[...]
        context_appear_diff = query_aware_context_diff[:, 0:self.context_dim, :] # only context appear needs back propagate

        query_aware_pool_appear_conv_diff = np.zeros(bottom[1].data[...].shape)
        for img in range(self.N):
            for word in range(self.T):
                if ['img, word', img, word] in self.hard_word_att_idx:
                    hard_word_att_idx = \
                        self.hard_word_att_idx[self.hard_word_att_idx.index(['img, word', img, word]) + 1]
                    if hard_word_att_idx.size == 1:
                        query_aware_pool_appear_conv_diff[img, :, word] = \
                            query_aware_pool_appear_conv_diff[img, :, word] \
                            + context_appear_diff[img, :, hard_word_att_idx]
                    else:
                        query_aware_pool_appear_conv_diff[img, :, word] = \
                            query_aware_pool_appear_conv_diff[img, :, word] \
                            + np.sum(context_appear_diff[img, :, hard_word_att_idx], axis=0)

        bottom[1].diff[...] = query_aware_pool_appear_conv_diff



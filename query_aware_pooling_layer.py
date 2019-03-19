from __future__ import division
import caffe
import numpy as np

class QueryAwarePoolingLayer(caffe.Layer):
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

        # query-aware appear pool for every word
        top[0].reshape(self.N,  self.context_dim,  self.T)
        # query-aware spatial position pool for every word
        top[1].reshape(self.N,  self.spatial_dim,  self.T)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward(self, bottom, top):

        # language attention for every image location
        language_att = bottom[0].data  # N*HW*T
        # features for every image location
        img_appear_feat = bottom[1].data  # N*C*H*W
        # img_spatial_features for every image location
        img_spatial_feat = bottom[2].data  # N*8*H*W

        img_feat = np.hstack((img_appear_feat, img_spatial_feat)) # N*(C+8)*H*W
        img_feat_dim = img_feat.shape[1]

        query_aware_pool = np.zeros((self.N, img_feat_dim, self.T))
        for img in range(self.N):
            language_att_cur = np.squeeze(language_att[img*self.HW: (img+1)*self.HW, :, :])  # language_att of current image
            img_feat_cur = img_feat[img, :, :, :]  # feature of current image
            img_feat_cur = np.reshape(img_feat_cur, (img_feat_dim, self.HW))

            for word in range(self.T):
                word_att = language_att_cur[:, word]
                hard_word_att_idx = np.squeeze(np.where(word_att >= self.key_word_thresh))
                if hard_word_att_idx.size > 0:
                    self.hard_word_att_idx.append(['img, word', img, word])
                    self.hard_word_att_idx.append(hard_word_att_idx)

                    img_feat_att = img_feat_cur[:, hard_word_att_idx]
                    if hard_word_att_idx.size == 1:
                        query_aware_pool[img, :, word] = img_feat_att
                    else:
                        query_aware_pool[img, :, word] = np.mean(img_feat_att, axis=1)

        top[0].data[...] = query_aware_pool[:, 0:self.context_dim, :]
        top[1].data[...] = query_aware_pool[:, self.context_dim:, :]

    def backward(self, top, propagate_down, bottom):
        query_aware_pool_appear_diff = top[0].diff[...]  # only context appear needs back propagate
        
        img_appear_feat_diff = np.zeros((self.N, bottom[1].data[...].shape[1], self.HW))
        for img in range(self.N):                                                                           
            for word in range(self.T):                                                                      
                if ['img, word', img, word] in self.hard_word_att_idx:                                      
                    hard_word_att_idx = \
                        self.hard_word_att_idx[self.hard_word_att_idx.index(['img, word', img, word]) + 1]

                    img_appear_feat_diff[img, :, hard_word_att_idx] = \
                        img_appear_feat_diff[img, :, hard_word_att_idx] \
                        + (1.0/hard_word_att_idx.size) * query_aware_pool_appear_diff[img, :, word]

        bottom[1].diff[...] = np.reshape(img_appear_feat_diff, (bottom[1].data[...].shape))
import tensorflow as tf
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

from helper import *
from batch_test import *

class NGCF(object):
    def __init__(self, data_config, pretrain_data):
        # 인자 세팅
        self.model_type = "ngcf"
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        
        self.pretrain_data = pretrain_data
        
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]

        self.n_fold = 100  # ?
        
        self.norm_adj = data_config["norm_adj"]
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr
        
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        
        self.weight_size = eval(args.layer_size)  # ?
        self.n_layers = len(self.weight_size)  # [64, 64, 64]
        
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)
        
        self.regs = eval(args.regs)  # ?
        self.decay = self.regs[0]

        self.verbose = args.verbose
        
        # 인풋과 데이터, 드롭아웃을 위한 Placeholder 생성
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flat
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        # 모델 파라미터 생성 (가중치 초기화)
        self.weights = self._init_weights()

        # 메시지 패싱 알고리즘 이용한 그래프 기반의 표상 학습 구현
        if self.alg_type in ["ngcf"]:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        
        elif self.alg_type in ["gcn"]:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()
        
        elif self.alg_type in ["gcmc"]:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()
                        
        # 배치 내의 유저-아이템 페어에 대한 최종 표상 생성
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        
    def _init_weights(self):
        pass
    
    def _split_A_hat(self, X):
        pass
    
    def _split_A_hat_node_dropout(self, X):
        pass
    
    def _create_ngcf_embed(self):
        pass
    
    def _create_gcn_embed(self):
        pass
    
    def _create_gcmc_embed(self):
        pass
    
    def create_bpr_loss(self, users, pos_items, neg_items):
        pass
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        pass
    
    def _dropout_sparse(self, X, keep_prob, n_nonzero,_elems):
        pass
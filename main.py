#!/usr/bin/env python3

# Code developed based on https://jianxinma.github.io/disentangle-recsys.html
# We develop a new constraned MOO algorithm

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import csv
import os
import shutil
import sys
import time
from tqdm import tqdm
import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing
import tensorflow as tf
from scipy import sparse
from tensorflow.contrib.distributions import RelaxedOneHotCategorical
from tensorflow.contrib.layers import apply_regularization, l2_regularizer, l1_regularizer
from metrics import ndcg_binary_at_k_batch, precision_recall_at_k_batch
from dataloader import load_data, load_item_cate, load_train_data, load_tr_te_data, sampler
from min_norm_solvers_numpy import MinNormSolver, gradient_normalizers
import scipy.sparse as sp
from metrics import hit_precision_recall_ndcg_k

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3' 
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

ARG = argparse.ArgumentParser()
ARG.add_argument('--data', type=str, default='./data/netflix',
                 help='./data/simulated,./data/ml-100k, ./data/ml-1m, ./data/netflix')
ARG.add_argument('--mode', type=str, default='tst',
                 help='trn/tst/vad, for training/testing/validation.')
ARG.add_argument('--MOO', type=str, default=True,
                 help='original or multi-group optimization')

ARG.add_argument('--lagrangian_method', type=str, default=False,
                 help='original or multi-group optimization')
ARG.add_argument('--logdir', type=str, default='./runs/')
ARG.add_argument('--seed', type=int, default=98765,
                 help='Random seed. Ignored if < 0.')
ARG.add_argument('--epoch', type=int, default=300,
                 help='Number of training epochs.') 
ARG.add_argument('--batch', type=int, default=500,
                 help='Training batch size.')
ARG.add_argument('--lr', type=float, default=1e-3,  # 5e-4
                 help='Initial learning rate.')
ARG.add_argument('--lr_lagrange_factor', type=float, default=1e-4,
                 help='Initial learning rate.')
ARG.add_argument('--rg', type=float, default=0.0,
                 help='L2 regularization.')
ARG.add_argument('--keep', type=float, default=0.5,
                 help='Keep probability for dropout, in (0,1].')
ARG.add_argument('--beta', type=float, default=1.0,
                 help='Strength of disentanglement, in (0,oo).')
ARG.add_argument('--tau', type=float, default=0.1,
                 help='Temperature of sigmoid/softmax, in (0,oo).')
ARG.add_argument('--std', type=float, default=0.075,
                 help='Standard deviation of the Gaussian prior.')
ARG.add_argument('--kfac', type=int, default=1,
                 help='Number of facets (macro concepts).')
ARG.add_argument('--dfac', type=int, default=100,
                 help='Dimension of each facet.')
ARG.add_argument('--nogb', action='store_true', default=False,
                 help='Disable Gumbel-Softmax sampling.')
ARG.add_argument('--normalization_type', type=str, default='none',
                 help='loss, l2, loss+, none')
ARG.add_argument('--c', type=float, default=50,
                 help='bound')
ARG = ARG.parse_args()

if ARG.seed < 0:
    ARG.seed = int(time.time())

batch_size_vad = batch_size_test = 3*ARG.batch

def set_rng_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def log_factorial(number):
    val = 0
    for i in range(1, number+1):
        val += np.log(i)
    return val

class MyVAE(object):
    def __init__(self, num_items):
        kfac, dfac = ARG.kfac, ARG.dfac
        batch_size = ARG.batch
        self.lam = ARG.rg
        self.lr = ARG.lr
        self.lr_lagrange_factor = ARG.lr_lagrange_factor
        self.random_seed = ARG.seed
        self.c = ARG.c

        self.n_items = num_items
        print('num_items: ', num_items, 'dfac: ',dfac, 'kfac: ', kfac)

        # The first fc layer of the encoder Q is the context embedding table.
        self.lagrange_factor = tf.Variable(0.0, name="lagrange_factors")
        # self.lagrange_factor = tf.Variable(tf.zeros(group_num), name="lagrange_factors")

        self.q_dims = [num_items, dfac, dfac]
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(
                zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2  # mu & var
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            bias_key = "bias_q_{}".format(i + 1)
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

        self.items = tf.get_variable(
            name="items", shape=[num_items, dfac],
            initializer=tf.contrib.layers.xavier_initializer(
                seed=self.random_seed))

        self.cores = tf.get_variable(
            name="cores", shape=[kfac, dfac],
            initializer=tf.contrib.layers.xavier_initializer(
                seed=self.random_seed))

        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, num_items])
        self.keep_prob_ph = tf.placeholder_with_default(1., shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)
        self.tsk_weights_ph = tf.placeholder(dtype=tf.float32, shape=[None,])
        self.group_ph = tf.placeholder(dtype=tf.float32, shape=[None,])
        
    def build_graph(self, save_emb=False):
        if save_emb:
            saver, facets_list = self.forward_pass(save_emb=True)
            return saver, facets_list, self.items, self.cores

        saver, logits, recon_loss_users, kl_users = self.forward_pass(save_emb=False)

        reg_var = apply_regularization(
            l2_regularizer(self.lam),
            self.weights_q + [self.items, self.cores])
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        recon_loss_group_list = self.multi_group_loss(recon_loss_users)
        kl_group_list = self.multi_group_loss(kl_users)
        multi_loss_list = [recon_loss_group_list[i]+ self.anneal_ph * kl_group_list[i] for i in range(group_num)]

        recon_prob_users = tf.reduce_sum(tf.nn.softmax(logits) * self.input_ph, axis=-1) 
        fairness_violation = tf.nn.relu(self.fair_loss(multi_loss_list) - self.c)  #

        share_variables = self.weights_q + self.biases_q #+ [self.items, self.cores]
        grad_task_list = []
        for loss in multi_loss_list:
            grad_task_list.append(tf.gradients(loss, share_variables))

        specific_loss = tf.reduce_mean(recon_loss_users) + self.anneal_ph * tf.reduce_mean(kl_users) + reg_var
        proxy_loss = reg_var
        proxy_loss = proxy_loss + tf.reduce_sum(tf.stack(multi_loss_list, axis=0)*self.tsk_weights_ph)

        lagrange_term = self.lagrange_factor * fairness_violation

        train_op_lagrange = tf.train.AdamOptimizer(learning_rate=self.lr_lagrange_factor, name='Adam').minimize(-lagrange_term, 
                                                var_list=[self.lagrange_factor])

        train_op_share = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam').minimize(proxy_loss + lagrange_term, 
                                                var_list=share_variables)
        
        specific_loss = specific_loss + tf.reduce_sum(tf.stack(multi_loss_list, axis=0))
        train_op_specific = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam').minimize(specific_loss + lagrange_term, 
                                                var_list=[self.items, self.cores])
        #train_op_objective = tf.train.AdamOptimizer(learning_rate=self.lr, name='Adam').minimize(proxy_loss)
        tf.summary.scalar('trn/proxy_loss', proxy_loss)
        merged = tf.summary.merge_all()

        return saver, logits, [train_op_specific, train_op_share, train_op_lagrange], grad_task_list, multi_loss_list, merged, [lagrange_term]

    def forward_pass(self, save_emb):
        
        cores = tf.nn.l2_normalize(self.cores, axis=1)
        items = tf.nn.l2_normalize(self.items, axis=1)
        cates_logits = tf.matmul(items, cores, transpose_b=True) / ARG.tau 

        if ARG.nogb:
            cates = tf.nn.softmax(cates_logits, axis=1)
        else: 
            cates_dist = RelaxedOneHotCategorical(1, cates_logits)
            cates_sample = cates_dist.sample()
            cates_mode = tf.nn.softmax(cates_logits, axis=1) # 进行softmax�?  num_item*num_core
            cates = (self.is_training_ph * cates_sample +
                     (1 - self.is_training_ph) * cates_mode)
        '''
        cores_list = []
        cates_max = tf.reduce_max(cates_mode, axis=1)
        for k in range(ARG.kfac):
                # cates_mode_k = tf.reshape(cates_mode[:, k], (-1,1))
                # cates_max_k = tf.gather(cates_max, tf.where(tf.equal(cates_max,cates_mode[:, k])))
                cates_max_k = tf.cast(tf.equal(cates_max,cates_mode[:, k]), tf.float32)
                cates_max_k = tf.reshape(cates_max_k,(-1,1))
                cores_list.append(tf.reshape(tf.reduce_sum(self.items * cates_max_k,axis=0), (1,-1)))
        #for k in range(ARG.kfac):

        #    cates_mode_k = tf.reshape(cates_mode[:, k], (-1,1))
        #    cores_list.append(tf.reshape(tf.div(tf.reduce_sum(self.items * cates_mode_k,axis=0),tf.reduce_sum(cates_mode_k)), (1,-1)))

        cores_list = tf.concat(cores_list, axis=0)
        self.cores=tf.assign(self.cores, cores_list)
        '''
        z_list = []
        probs, kl = None, None 
        for k in range(ARG.kfac):
            cates_k = tf.reshape(cates[:, k], (1, -1))
            # q-network
            x_k = self.input_ph * cates_k
            mu_k, std_k, kl_k_users = self.q_graph_k(x_k)
            epsilon = tf.random_normal(tf.shape(std_k))
            z_k = mu_k + self.is_training_ph * epsilon * std_k
            if k==0:
                kl_users = kl_k_users
            else:
                kl_users += kl_k_users 
            # kl_group_tf = (kl_k_group_tf if (kl_group_tf is None) else (kl_group_tf + kl_k_group_tf))  将list转化为tensor的化
            if save_emb:
                z_list.append(z_k)
            # p-network
            z_k = tf.nn.l2_normalize(z_k, axis=1)
            logits_k = tf.matmul(z_k, items, transpose_b=True)
            probs_k = tf.exp(logits_k / ARG.tau) 
            probs_k = probs_k * cates_k  #(num_users * num_items)*(1, num_items)
            probs = (probs_k if (probs is None) else (probs + probs_k))
        logits = tf.log(probs)  #(-10, 10+)

        softmax_logits = tf.nn.softmax(logits)
        #logits is a vector of user's preference score towards all items
        recon_loss_users = tf.reduce_sum(- tf.log(softmax_logits) * self.input_ph, axis=-1)

        if save_emb:
            return tf.train.Saver(), z_list
        return tf.train.Saver(), logits, recon_loss_users, kl_users

    def q_graph_k(self, x):
        mu_q, std_q, kl = None, None, None
        h = tf.nn.l2_normalize(x, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w, a_is_sparse=(i == 0)) + b
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]  # a^k_u
                mu_q = tf.nn.l2_normalize(mu_q, axis=1)
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:] # b^k_u
                std0 = ARG.std
                std_q = tf.exp(0.5 * lnvarq_sub_lnvar0) * std0
                # Trick: KL is constant w.r.t. to mu_q after we normalize mu_q.
                kl_users = tf.reduce_sum(0.5 * (-lnvarq_sub_lnvar0 + tf.exp(lnvarq_sub_lnvar0) - 1.), axis=1)

        return mu_q, std_q, kl_users

    def multi_group_loss(self, loss_users):
        loss_group_list = []
        for i in range(group_num):
            loss_i = tf.reduce_mean(tf.gather(loss_users, tf.where(tf.equal(self.group_ph, i))))   
            loss_group_list.append(loss_i)      
        return loss_group_list

    def fair_loss(self, loss_group_list):
        loss_mean = tf.reduce_mean(tf.stack(loss_group_list, axis=0))
        for i, loss in enumerate(loss_group_list):
            if i==0:
                #fair_constraints = tf.cast(tf.greater(loss, 1.0*loss_mean), tf.float32) * loss 
                fair_constraints = tf.nn.relu(loss - loss_mean)
            else:
                #fair_constraints += tf.cast(tf.greater(loss, 1.0*loss_mean), tf.float32) * loss
                fair_constraints += tf.nn.relu(loss - loss_mean)
        return fair_constraints



def main_train_vad(ARG, train_data, train_group_dict, vad_data, vad_group_dict, validation=True, best_epoch=None):#, Nu_list):
    set_rng_seed(ARG.seed)
    if not validation:
        train_data = train_data + vad_data
        for key in train_group_dict:
            train_group_dict[key] = list(set(train_group_dict[key]).union(set(vad_group_dict[key])))

    n = train_data.shape[0] # train-users, train_data是一个csr_matrix
    n_items = train_data.shape[1]
    idxlist = list(range(n))
    n_vad = vad_data.shape[0]

    num_batches = int(np.ceil(float(n) / ARG.batch))
    total_anneal_steps = num_batches 

    tf.reset_default_graph()
    vae = MyVAE(n_items)
    saver, logits_var, train_op_list, grad_task_list, multi_loss_list, merged_var, lagrange_list = vae.build_graph()

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if validation:
            epochs = ARG.epoch
        else:
            epochs = int(1.2 * best_epoch)

        best_recall = 0.0
        best_epoch = 0
        best_grad_norm = np.inf
        update_count = 0.0 
        scale=np.array([1/num_tsk]*num_tsk)

        for epoch in tqdm(range(epochs)):
            for group in train_group_dict:
                random.shuffle(train_group_dict[group])
            st_idx_dict = dict((group, 0) for group in range(group_num))

            recon_train_loss_list, train_r30_list = [],[]
            r30_train_group_list = []
            x_group_list =[]

            for bnum, st_idx in enumerate(range(0, n, ARG.batch)):
                x_group, x, st_idx_dict = sampler(train_data, st_idx_dict, ARG.batch, train_group_dict)

                train_set = sp.lil_matrix(x).rows
                max_train_count = 0
                vad_item = [[] for i in range(len(train_set))]

                if sparse.isspmatrix(x):
                    x = x.toarray()
                x = x.astype('float32')

                if total_anneal_steps > 0:
                    anneal = min(ARG.beta,
                                 1. * update_count / total_anneal_steps)
                else:
                    anneal = ARG.beta

                feed_dict = {vae.input_ph: x,
                             vae.keep_prob_ph: ARG.keep,
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1,
                             vae.group_ph:(np.array(x_group)).astype('float32')}
                results = sess.run([train_op_list[0]]+grad_task_list+multi_loss_list, feed_dict=feed_dict)  # python 列表相加=列表相拼�?
                grads = results[1:num_tsk+1]
                #recon_train_loss_list.append(results[num_tsk:(num_tsk+group_num)])
                recon_train_loss_list.append(results[1+num_tsk:])
                '''
                gn = gradient_normalizers(grads, results[num_tsk:], ARG.normalization_type)
                for t in range(num_tsk):
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]
                '''
                if ARG.MOO:
                    for t, grad_t in enumerate(grads):
                        grads[t]= np.hstack((g.reshape(-1)) for g in grad_t)
                    scale, min_norm = MinNormSolver.find_min_norm_element(grads)
                    scale = np.minimum(1.0, scale + 0.2)
                    
                    if min_norm < 5:
                        saver.save(sess, '{}/chkpt'.format(LOG_DIR))
                        return best_recall

                feed_dict = {vae.input_ph: x,
                             vae.keep_prob_ph: ARG.keep,
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1,
                             vae.tsk_weights_ph:scale,
                             vae.group_ph:(np.array(x_group)).astype('float32')}
                results = sess.run([logits_var, train_op_list[1]], feed_dict=feed_dict)
                pred_val = results[0]

                if ARG.lagrangian_method:
                    results = sess.run([train_op_list[2]] + lagrange_list, feed_dict=feed_dict)
                    # print('lagrange_list', results[1:])
                
                
                _, _, recall_k, _ = hit_precision_recall_ndcg_k(vad_item, train_set, np.squeeze(np.array(pred_val)), 
                                                              max_train_count, k=30, ranked_tag=False)
                train_r30_list.extend(recall_k)
                x_group_list.extend(x_group)


                if bnum % 50 == 0:
                    summary_train = sess.run(merged_var, feed_dict=feed_dict)
                    summary_writer.add_summary(
                        summary_train,
                        global_step=epoch * num_batches + bnum)
                update_count += 1

            x_group_list = np.array(x_group_list)
            train_r30_list = np.array(train_r30_list)
            recon_train_loss_list = np.array(recon_train_loss_list)

           
            if validation:

                r30_list = []
                x_group_list = []
                recon_vad_loss_list = []
                r30_vad_group_list = []


                st_idx_dict = dict((group, 0) for group in range(group_num))
                for bnum, st_idx in enumerate(range(0, n_vad, batch_size_vad)):

                    x_group, x, x_te, st_idx_dict = sampler(train_data, st_idx_dict, batch_size_vad, vad_group_dict, vad_data)              

                    train_set = sp.lil_matrix(x).rows
                    max_train_count = np.max([len(t) for t in train_set])
                    vad_item = sp.lil_matrix(x_te).rows

                    if sparse.isspmatrix(x):
                        x = x.toarray()
                    x = x.astype('float32')
                    results = sess.run([logits_var, multi_loss_list], feed_dict={vae.input_ph: x, 
                                                                                vae.is_training_ph: 0,
                                                                                vae.group_ph:(np.array(x_group)).astype('float32')})

                    pred_val = results[0]
                    recon_vad_loss_list.append(results[1][:group_num])

                    pred_val[x.nonzero()] = -np.inf
                    _, _, recall_k, _ = hit_precision_recall_ndcg_k(train_set, vad_item, np.squeeze(np.array(pred_val)), 
                                                                                            max_train_count, k=30, ranked_tag=False)

                    r30_list.extend(recall_k)
                    x_group_list.extend(x_group)

                r30_list = np.array(r30_list)
                recon_vad_loss_list = np.array(recon_vad_loss_list)
                recall = r30_list.mean()
                x_group_list = np.array(x_group_list)


                for i in range(group_num):
                    train_r30_tmp = (train_r30_list[x_group_list==i]).mean()
                    if not np.isnan(train_r30_tmp):
                        r30_train_group_list.append(train_r30_tmp)

                    r30_tmp = (r30_list[x_group_list==i]).mean()
                    if not np.isnan(r30_tmp):
                        r30_vad_group_list.append(r30_tmp)

            # ==================================================================================================
            else:
                recall = train_r30_list.mean()
                for i in range(group_num):
                    train_r30_tmp = (train_r30_list[x_group_list==i]).mean()
                    if not np.isnan(train_r30_tmp):
                        r30_train_group_list.append(train_r30_tmp)
                print('train_negelbo: ', recon_train_loss_list.mean(0), '\ntrain_recall: ', r30_train_group_list)

            if recall > best_recall:
                best_epoch = epoch
                saver.save(sess, '{}/chkpt'.format(LOG_DIR))
                best_recall = recall

    return best_epoch


def main_tst(ARG,train_data, vad_data, tst_data, tst_group_dict, report_r20=False):
    set_rng_seed(ARG.seed)
    n_test = tst_data.shape[0]
    n_items = tst_data.shape[1]
    idxlist_test = list(range(n_test))

    tf.reset_default_graph()
    vae = MyVAE(n_items)
    saver, logits_var, _, _, multi_loss_list, _, lagrange_list = vae.build_graph()

    h30_list, r30_list = [], []
    h20_list, r20_list = [], []

    x_group_list = []
    tst_cnt_array = np.zeros(group_num)

    # train_data = sp.lil_matrix(train_data)
    # tst_data = sp.lil_matrix(tst_data)
    with tf.Session() as sess:
        saver.restore(sess, '{}/chkpt'.format(LOG_DIR))

        st_idx_dict = dict((group, 0) for group in range(group_num))
        tst_cnt = 0        
        recon_tst_loss_list = []
        for bnum, st_idx in enumerate(range(0, n_test, batch_size_test)):

            x_group, x, x_te, x_vad, st_idx_dict = sampler(train_data, st_idx_dict, batch_size_test, tst_group_dict, tst_data, vad_data)
            
            train_set = sp.lil_matrix(x).rows
            if 'book-crossing' in ARG.data:
                x_vad = None
            else:
                x_vad = sp.lil_matrix(x_vad).rows
            max_train_count = np.max([len(t) for t in train_set])
            tst_item = sp.lil_matrix(x_te).rows


            if sparse.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')
            results = sess.run([logits_var, multi_loss_list], feed_dict={vae.input_ph: x,
                                                       vae.is_training_ph: 0,
                                                       vae.group_ph:(np.array(x_group)).astype('float32')})
            pred_val = results[0]
            # print('pred_val', np.max(pred_val), np.min(pred_val))
            pred_val[x.nonzero()] = -np.inf
            recon_tst_loss_list.append(results[1])

            hits_tmp, _, recall_tmp, _= hit_precision_recall_ndcg_k(train_set, tst_item, np.squeeze(np.array(pred_val)), 
                                                                                        max_train_count, k=30, ranked_tag=False, vad_set_batch=x_vad)
            h30_list.extend(hits_tmp)
            r30_list.extend(recall_tmp)

            hits_tmp, _, recall_tmp, _ = hit_precision_recall_ndcg_k(train_set, tst_item, np.squeeze(np.array(pred_val)), 
                                                                                        max_train_count, k=20, ranked_tag=False, vad_set_batch=x_vad)
            h20_list.extend(hits_tmp)
            r20_list.extend(recall_tmp)
            
            tst_cnt += x_te.count_nonzero()

            for i in range(group_num):
                tst_cnt_array[i] += np.sum([len(l) for l in tst_item[np.array(x_group)==i]])

            x_group_list.extend(x_group)

        recon_tst_loss_list = np.array(recon_tst_loss_list).mean(0)
        h30_list = np.array(h30_list)
        r30_list = np.array(r30_list)
        h20_list = np.array(h20_list)
        r20_list = np.array(r20_list)


        x_group_list = np.array(x_group_list)


        r20_group_list, r30_group_list = [],[]
        hit20_group_list, hit30_group_list = [], []

        for i in range(group_num):
            r20_tmp = (r20_list[x_group_list==i]).mean()
            if not np.isnan(r20_tmp):
                r20_group_list.append(r20_tmp)

            hit20_tmp = (h20_list[x_group_list==i]).sum() / tst_cnt_array[i]
            if not np.isnan(hit20_tmp):
                hit20_group_list.append(hit20_tmp)
            
            r30_tmp = (r30_list[x_group_list==i]).mean()
            if not np.isnan(r30_tmp):
                r30_group_list.append(r30_tmp)

            h30_tmp = (h30_list[x_group_list==i]).sum() / tst_cnt_array[i]
            if not np.isnan(h30_tmp):
                hit30_group_list.append(h30_tmp)

        recall20_diff = np.std(np.array(r20_group_list))
        hit20_diff = np.std(np.array(hit20_group_list))
        recall30_diff = np.std(np.array(r30_group_list))
        hit30_diff = np.std(np.array(hit30_group_list))

       
        print('===================================================================================================')
        print('recon_tst_loss_list', recon_tst_loss_list)
        print('r30_group_list', r30_group_list, '\nhit30_group_list', hit30_group_list)
        print('r20_group_list', r20_group_list, '\nhit20_group_list', hit20_group_list)

        print("Test HR@20=%.5f" % (
            h20_list.sum()/tst_cnt),
              file=sys.stderr)
        print("Test Recall@20=%.5f (%.5f)" % (
            r20_list.mean(), np.std(r20_list) / np.sqrt(len(r20_list))),
              file=sys.stderr)
        print("Test HR@30=%.5f " % (
            h30_list.sum()/tst_cnt),
              file=sys.stderr)
        print("Test Recall@30=%.5f (%.5f)" % (
            r30_list.mean(), np.std(r30_list) / np.sqrt(len(r30_list))),
              file=sys.stderr)
        
        print("Test difference betwen groups are: \n Recall@20=%.5f, hit@20=%.5f, Recall@30=%.5f, hit@30=%.5f,"%(recall20_diff, 
                                                                            hit20_diff, recall30_diff, hit30_diff))
    
    f = open(ARG.data+'/hyper_search.txt', 'a')
    f.write('Recall20: %.5f'% r20_list.mean()+'\t')
    f.write('HitRate20: %.5f'% (h20_list.sum()/tst_cnt) +'\t')
    f.write('Recall30: %.5f'%r30_list.mean()+'\t')
    f.write('HitRate30: %.5f'% (h30_list.sum()/tst_cnt)+'\t')

    f.write('Recall20-std: %.5f'% recall20_diff+'\t')
    f.write('HitRate20-std: %.5f'% hit20_diff+'\t')
    f.write('Recall30-std: %.5f'% recall30_diff+'\t')
    f.write('HitRate30-std: %.5f'% hit30_diff+'\t')


    f.write('beta:'+str(ARG.beta)+'\tdfac:'+str(ARG.dfac)+'\tkfac:'+str(ARG.kfac)+'\tkeep:'+str(ARG.keep))
    f.write('\n')
    f.close()
    

    if report_r20:
        return r20_list.mean()
    return r20_list.mean()



if __name__ == '__main__':
    (n_items, n_users, train_data, train_group_dict, vad_data, vad_group_dict,
            tst_data,  tst_group_dict)= load_data(ARG.data)
    print('finishing loading data', '%d users and %d items'%(n_users, n_items))

    global group_num
    global num_tsk
    group_num = len(train_group_dict)
    num_tsk = group_num

    val, tst = 0, 0
    best_epoch = int(ARG.epoch/1.2)

    global LOG_DIR
    LOG_DIR = os.path.basename(sys.argv[0])[:-3] + '%s-%sMOO-%dT-%glr-%dk-%dd-%ds-%slag' % (
    ARG.data.replace('/', '_'), ARG.MOO, ARG.epoch, ARG.lr, ARG.kfac, ARG.dfac, ARG.seed, ARG.lagrangian_method)
    if ARG.nogb:
        LOG_DIR += '-nogb'
    LOG_DIR = os.path.join(ARG.logdir, LOG_DIR)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if ARG.mode in ('vad'):
        best_epoch = main_train_vad(ARG,train_data, train_group_dict, vad_data, vad_group_dict, validation=True)#, Nu_list)
        print('======= validation finished, the best epoch is {} ======='.format(best_epoch))
    if ARG.mode in ('trn'):
        best_epoch = main_train_vad(ARG,train_data, train_group_dict, vad_data, vad_group_dict, validation=False, best_epoch = best_epoch)
        print('======= training finished, the best epoch is {} ======='.format(best_epoch))

    if ARG.mode in ('trn', 'vad', 'tst'):
        tst = main_tst(ARG, train_data, vad_data, tst_data, tst_group_dict)  # 其实不用vad�?把train和vad合并成train就可以了

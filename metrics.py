import numpy as np
import bottleneck as bn
import math,random

def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = x_pred.shape[0]
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1) # shape和x_pred一样
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]] # 所有用户的topk item同时挑出来
    idx_part = np.argsort(-topk_part, axis=1) 
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0
    return ndcg


def precision_recall_at_k_batch(x_pred, heldout_batch, k=100, observe_fair=False, attr_indicator_list=None):
    batch_users = x_pred.shape[0]

    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()

    ranked_list = x_pred_binary

    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / x_true_binary.sum(axis=1)
    precision = tmp / k
    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    F1 = 2 * recall * precision / (precision + recall)
    F1[np.isnan(F1)] = 0
    return precision, recall, F1

def update_threshold(x_pred, id_onehots_ph, threshold_ph, k=100):
    batch_users = x_pred.shape[0]
    idx = bn.argpartition(-x_pred, k, axis=1)
    #epsion = 1e-10
    #threshold_ph_batch = x_pred[:, idx[:, k]]-epsion
    #print('shape(threshold_ph_batch)', threshold_ph_batch.shape)
    threshold_ph[np.nonzero(id_onehots_ph)[1]] = x_pred[np.arange(batch_users), idx[:, k]].reshape(-1,1)
    #threshold_ph = np.dot(threshold_ph.T, id_onehots_ph.toarray())
    return threshold_ph



def AP(ranked_list, ground_truth):
    """Compute the average precision (AP) of a list of ranked items
    """
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0

def mrr(gt_items, ranked_pred_items):
    for index,item in enumerate(ranked_pred_items):
        if item in gt_items:
            return 1/index

def hit(gt_items, pred_items):  # HR为所有用户的hits/所有用户的grounf truth总个数
    count = 0
    for item in pred_items:
        if item in gt_items:
            count += 1
    return count


def auc(label, prob): # prob 为预测为正的概率
    precision, recall, _thresholds = metrics.precision_recall_curve(label, prob)
    area = metrics.auc(recall, precision)
    return area
# sklearn
# precision, recall, _thresholds = metrics.precision_recall_curve(label, prob)
# area = metrics.auc(recall, precision)
# return area


# area = metrics.roc_auc_score(label, prob)
# return area

def hit_precision_recall_ndcg_k(train_set_batch, test_set_batch, pred_scores_batch, max_train_count, k=20, ranked_tag=False, vad_set_batch=None):
    recall_k,precision_k,NDCG_k, hits_list = [],[],[],[]

    
    if not ranked_tag:
        batch_users = pred_scores_batch.shape[0]
        idx_topk_part = bn.argpartition(-pred_scores_batch, k+max_train_count, axis=1)
        
        topk_part = pred_scores_batch[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :(k+max_train_count)]] # index 1318 is out of bounds for axis 0 with size 498
        idx_part = np.argsort(-topk_part, axis=1) 

        top_items = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    else:
        top_items = pred_scores_batch
    if vad_set_batch is None:
        for train_set, test_set, ranked in zip(train_set_batch, test_set_batch, top_items):

            n_k = k if len(test_set) > k else len(test_set) # n_k = min(k, len(test_k))

            _idcg,_dcg = 0,0
            for pos in range(n_k):
                _idcg += 1.0 / math.log(pos + 2,2)

            tops_sub_train = []
            n_top_items = 0
            for val in ranked:
                if val in train_set:
                    continue
                else:
                    tops_sub_train.append(val)

                n_top_items += 1

                if n_top_items >= k: # 控制topK个item是从用户没交互过的商品中选的
                    break
            hits_set = [(idx,itemID) for idx, itemID in enumerate(tops_sub_train) if itemID in test_set]
            cnt_hits = len(hits_set)

            for idx in range(cnt_hits):
                _dcg += 1.0 /math.log(hits_set[idx][0] + 2,2)
            precision_k.append(float(cnt_hits / k))
            recall_k.append(float(cnt_hits / len(test_set)))
            NDCG_k.append(float(_dcg / _idcg))
            hits_list.append(cnt_hits)
    else: 
        for train_set, test_set, ranked, vad_set in zip(train_set_batch, test_set_batch, top_items, vad_set_batch):

            n_k = k if len(test_set) > k else len(test_set) # n_k = min(k, len(test_k))

            _idcg,_dcg = 0,0
            for pos in range(n_k):
                _idcg += 1.0 / math.log(pos + 2,2)

            tops_sub_train = []
            n_top_items = 0
            for val in ranked:
                if val in train_set or val in vad_set:
                    continue
                else:
                    tops_sub_train.append(val)

                n_top_items += 1

                if n_top_items >= k: # 控制topK个item是从用户没交互过的商品中选的
                    break
            hits_set = [(idx,itemID) for idx, itemID in enumerate(tops_sub_train) if itemID in test_set]
            cnt_hits = len(hits_set)

            for idx in range(cnt_hits):
                _dcg += 1.0 /math.log(hits_set[idx][0] + 2,2)
            precision_k.append(float(cnt_hits / k))
            recall_k.append(float(cnt_hits / len(test_set)))
            NDCG_k.append(float(_dcg / _idcg))
            hits_list.append(cnt_hits)

    return hits_list, precision_k, recall_k, NDCG_k


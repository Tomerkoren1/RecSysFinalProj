import numpy as np

def MRR_for_user(user_true, user_pred, lower_bound=1, upper_bound=5, top_n=10, threshold=4):
    # get all itmes that not rated and remove them from prediactions
    counter = 1
    user_actual_rating = user_true[user_true.nonzero()]
    user_pred = user_pred[user_true.nonzero()]
    amount_to_recommend = min(top_n, len(user_actual_rating))
    user_pred_sorted_idxs = user_pred.argsort()[::-1][:amount_to_recommend]
    for idx in user_pred_sorted_idxs:
        if user_actual_rating[idx] >= threshold:
            return 1/counter
        counter += 1
    return 0

def NDCG_for_user(user_true,user_pred,lower_bound=1,upper_bound=5,top_n=10):
    user_actual_rating = user_true[user_true.nonzero()]
    user_pred = user_pred[user_true.nonzero()]
    amount_to_recommend = min(top_n,len(user_actual_rating))
    user_pred_sorted_idxs = user_pred.argsort()[::-1][:amount_to_recommend]
    rel = []
    for idx in user_pred_sorted_idxs:
        rel.append(user_actual_rating[idx])
    dcg_p = DCG(rel,top_n)
    rel.sort(reverse = True)
    Idcg_p = DCG(rel,top_n)
    return dcg_p/Idcg_p if Idcg_p > 0 else 0

def DCG(rel,n):
    total = 0
    for i in range(1,len(rel)+1):
        calc = rel[i-1]/np.log2(i+1)
        total +=calc
    return total
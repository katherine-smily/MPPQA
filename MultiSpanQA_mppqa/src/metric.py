from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

def hit_metric(total_preds,total_labels):
    hit_num=0
    total_sum=0
    for preds,labels in zip(total_preds,total_labels):
        total_sum+=len(labels)
        hit_num+=len(set(preds).intersection(set(labels)))
    print(f"hit_num/total_sum={hit_num}/{total_sum}")
    return hit_num/total_sum

def accuracy_metric(preds, labels, pos_prob, isbaseline=False):
    
    assert len(preds) == len(labels)
    lr_precision, lr_recall = 0, 0
    if not isbaseline:
        lr_precision, lr_recall, _ = precision_recall_curve(labels, pos_prob)
        
    out = {'accuracy': accuracy_score(preds, labels),
           'precision': precision_score(labels, preds),
           'recall': recall_score(labels, preds),
           'f1': f1_score(labels, preds),
           'classification_report':classification_report(labels, preds),
           'roc_auc_score':roc_auc_score(labels,pos_prob)  if not isbaseline else 0,
           "cohen":cohen_kappa_score(labels,preds),
           "pr_auc":auc(lr_recall, lr_precision) if not isbaseline else 0
          }
    return out


def precision_recall_f1_metric(preds, labels, average_type="micro"):
    assert len(preds) == len(labels)

    out = {'accuracy': accuracy_score(preds, labels),
           'precision': precision_score(labels, preds, average=average_type),
           'recall': recall_score(labels, preds, average=average_type),
           'f1': f1_score(labels, preds, average=average_type)
           }
    report = classification_report(labels, preds)

    return out, report


# 第二种
def bi_multi_metric(bi_preds, bi_labels, multi_preds, multi_labels):
    """
       bi_preds = [0, 1, 1, 0]
       bi_labels = [1, 1, 1, 0]
       multi_preds =[1, 2, 3, 0]
       multi_labels = [1, 2, 4, 0]
    """
    # return  <跟你那边两个baseline的评价指标>
    cnt_cor_bi=sum([x*y for x,y in zip(bi_preds,bi_labels)])
    cnt_pred=sum(bi_preds)
    cnt_golden=sum(bi_labels)
    if cnt_pred!=0:
        prec_bi = cnt_cor_bi * 1. / cnt_pred
    else:
        prec_bi=0

    if cnt_golden!=0:
        recall_bi=cnt_cor_bi * 1. / cnt_golden
    else:
        recall_bi=0
    if prec_bi == 0 or recall_bi == 0:
        f1_bi = 0
    else:
        f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)

    bi_multi_preds= [x*y for x,y in zip(bi_preds,multi_preds)]
    bi_multi_labels= [x*y for x,y in zip(bi_labels,multi_labels)]
    bi_multi_res= [1 if bi_multi_preds[i]>0 and bi_multi_preds[i]==bi_multi_labels[i] else 0 for i in range(len(bi_multi_preds))]
    cnt_cor_multi=sum(bi_multi_res)
    cnt_pred,cnt_golden=0,0
    for i in range(len(bi_multi_preds)):
        if bi_preds[i]==1 and multi_preds[i]>0:
            cnt_pred+=1

    for i in range(len(bi_multi_labels)):
        if bi_multi_labels[i]!=0:
            cnt_golden+=1

    if cnt_pred!=0:
        prec_multi = cnt_cor_multi * 1. / cnt_pred
    else:
        prec_multi=0

    if cnt_golden!=0:
        recall_multi=cnt_cor_multi * 1. / cnt_golden
    else:
        recall_multi=0
    if prec_multi == 0 or recall_multi ==0:
        f1_multi = 0
    else:
        f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    # prec_bi, recall_bi = cnt_cor_bi * 1. / cnt_pred, cnt_cor_bi * 1. / cnt_golden
    # f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    # prec_multi, recall_multi = cnt_cor_multi * 1. / cnt_pred, cnt_cor_multi * 1. / cnt_golden
    # f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return f1_bi, f1_multi, prec_bi,recall_bi,prec_multi,recall_multi


def sent_tag_metric(sent_pred,sent_label):
    sent_tag_classify_report=classification_report(sent_label,sent_pred)
    return sent_tag_classify_report

# 不依赖于link predition
# def direct_precision_recall_f1_metric(multi_preds,multi_labels, type2id):
#     """
#        bi_preds = [0, 1, 1, 0]
#        bi_labels = [1, 1, 1, 0]
#        multi_preds =[1, 2, 3, 0]
#        multi_labels = [1, 2, 4, 0]
#     """
#     id2type = {value:key for key, value in type2id.items()}
#     # return  <跟你那边两个baseline的评价指标>
#     bi_preds=[1 if x>0 else 0 for x in multi_preds ]
#     bi_labels=[1 if x>0 else 0 for x in multi_labels]
#     # bi_preds,bi_labels=multi_preds&,multi_labels
#     cnt_cor_bi=sum([x*y for x,y in zip(bi_preds,bi_labels)])
#     cnt_pred=sum(bi_preds)
#     cnt_golden=sum(bi_labels)
#     if cnt_pred!=0:
#         prec_bi = cnt_cor_bi * 1. / cnt_pred
#     else:
#         prec_bi=0
#
#     if cnt_golden!=0:
#         recall_bi=cnt_cor_bi * 1. / cnt_golden
#     else:
#         recall_bi=0
#     if prec_bi!=0 and recall_bi!=0:
#         f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
#     else:
#         f1_bi=0
#     bi_multi_preds=[x*y for x,y in zip(bi_preds,multi_preds)]
#     bi_multi_labels=[x*y for x,y in zip(bi_labels,multi_labels)]
#     bi_multi_res=[1 if bi_multi_preds[i]>0 and bi_multi_preds[i]==bi_multi_labels[i] else 0 for i in range(len(bi_multi_preds)) ]
#     cnt_cor_multi=sum(bi_multi_res)
#     cnt_pred,cnt_golden=0,0
#     for i in range(len(bi_multi_preds)):
#         if bi_preds[i]==1 and multi_preds[i]>0:
#             cnt_pred+=1
#
#     for i in range(len(bi_multi_labels)):
#         if bi_multi_labels[i]!=0:
#             cnt_golden+=1
#
#     if cnt_pred!=0:
#         prec_multi = cnt_cor_multi * 1. / cnt_pred
#     else:
#         prec_multi=0
#
#     if cnt_golden!=0:
#         recall_multi=cnt_cor_multi * 1. / cnt_golden
#     else:
#         recall_multi=0
#     if prec_multi!=0 and recall_multi!=0:
#         f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
#     else:
#         f1_multi=0
#
#     precision_rate,recall_rate,f1_score=detail_precision_recall_f1_metric(bi_multi_preds,bi_multi_labels,type2id)
#
#     # return f1_bi, f1_multi,prec_bi,recall_bi,prec_multi,recall_multi
#     # result=[]
#     # result.append([prec_bi,recall_bi,f1_bi])
#     # result.append([prec_multi,recall_multi,f1_multi])
#     #
#     # for i in range(len(precision_rate)):
#     #     result.append([precision_rate[i],recall_rate[i],f1_score[i]])
#
#     result = {}
#     for i in range(len(precision_rate)):
#         result[id2type[i]] = {"P": precision_rate[i-1], "R": recall_rate[i-1], "F1": f1_score[i-1]}
#
#     return result

# 依赖于link prediction
def simul_precision_recall_f1_metric(bi_preds, bi_labels, multi_preds, multi_labels, type2id):
    """
       bi_preds = [0, 1, 1, 0]
       bi_labels = [1, 1, 1, 0]
       multi_preds =[1, 2, 3, 0]
       multi_labels = [1, 2, 4, 0]
    """
    id2type = {value:key for key, value in type2id.items()}

    # return  <跟你那边两个baseline的评价指标>
    cnt_cor_bi = sum([x * y for x, y in zip(bi_preds, bi_labels)])
    cnt_pred = sum(bi_preds)
    cnt_golden = sum(bi_labels)
    if cnt_pred != 0:
        prec_bi = cnt_cor_bi * 1. / cnt_pred
    else:
        prec_bi = 0

    if cnt_golden != 0:
        recall_bi = cnt_cor_bi * 1. / cnt_golden
    else:
        recall_bi = 0
    if prec_bi != 0 and recall_bi != 0:
        f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    else:
        f1_bi = 0
    bi_multi_preds = [x * y for x, y in zip(bi_preds, multi_preds)]
    bi_multi_labels = [x * y for x, y in zip(bi_labels, multi_labels)]
    bi_multi_res = [1 if bi_multi_preds[i] > 0 and bi_multi_preds[i] == bi_multi_labels[i] else 0 for i in
                    range(len(bi_multi_preds))]
    cnt_cor_multi = sum(bi_multi_res)
    cnt_pred, cnt_golden = 0, 0
    for i in range(len(bi_multi_preds)):
        if bi_preds[i] == 1 and multi_preds[i] > 0:
            cnt_pred += 1

    for i in range(len(bi_multi_labels)):
        if bi_multi_labels[i] != 0:
            cnt_golden += 1

    if cnt_pred != 0:
        prec_multi = cnt_cor_multi * 1. / cnt_pred
    else:
        prec_multi = 0

    if cnt_golden != 0:
        recall_multi = cnt_cor_multi * 1. / cnt_golden
    else:
        recall_multi = 0
    if prec_multi != 0 and recall_multi != 0:
        f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    else:
        f1_multi = 0

    precision_rate, recall_rate, f1_score = detail_precision_recall_f1_metric(bi_multi_preds, bi_multi_labels, type2id)

    # return f1_bi, f1_multi,prec_bi,recall_bi,prec_multi,recall_multi
    result = []
    result.append([prec_bi, recall_bi, f1_bi])
    result.append([prec_multi, recall_multi, f1_multi])

    for i in range(len(precision_rate)):
        result.append([precision_rate[i], recall_rate[i], f1_score[i]])

    result = {}
    for i in range(len(precision_rate)):
        # result[id2type[i]] = {"P": precision_rate[i-1], "R": recall_rate[i-1], "F1": f1_score[i-1]}
        result[id2type[i]] = {"P": precision_rate[i], "R": recall_rate[i], "F1": f1_score[i]}

    return result

def detail_precision_recall_f1_metric(bi_multi_preds, bi_multi_labels, type2id):
    '''
    multi_preds =[1, 2, 3, 0]
    multi_labels = [1, 0, 2, 0]
    '''

    # prediction_type = ["none","next_action", "sub_action", "supplement"]
    id2type = {}
    for key in type2id:
        id2type[type2id[key]] = key
    # print(id2type)
    TP_prediction = [0] * len(type2id)
    FN_prediction = [0] * len(type2id)
    FP_prediction = [0] * len(type2id)

    for i in range(len(bi_multi_preds)):
        if bi_multi_preds[i] == bi_multi_labels[i]:
            TP_prediction[bi_multi_preds[i]] += 1
        else:
            FP_prediction[bi_multi_preds[i]] += 1
    for i in range(len(bi_multi_labels)):
        if bi_multi_labels[i] != bi_multi_preds[i]:
            FN_prediction[bi_multi_labels[i]] += 1

    recall_rate = []
    precision_rate = []
    f1_score = []

    for i in range(len(TP_prediction)):
        sum = TP_prediction[i] + FN_prediction[i]
        if sum != 0:
            recall_rate.append(TP_prediction[i] / sum)
        else:
            recall_rate.append(0)

        sum = TP_prediction[i] + FP_prediction[i]
        if sum != 0:
            precision_rate.append(TP_prediction[i] / sum)
        else:
            precision_rate.append(0)

        sum = recall_rate[i] + precision_rate[i]
        if sum != 0:
            f1_score.append(2 * recall_rate[i] * precision_rate[i] / (sum))
        else:
            f1_score.append(0)

    # for i in range(len(type2id)):
    #     # print(i)
    #     print(id2type[i],
    #           ":\nprecision rate:", precision_rate[i],
    #           "\nrecall rate:", recall_rate[i],
    #           "\nf1 score:", f1_score[i], "\n\n")
    # print(bi_multi_preds, bi_multi_labels)
    # print(TP_prediction,FN_prediction,FP_prediction)
    # print(prediction_type,precision_rate,recall_rate,f1_score)
    # print(precision_rate, recall_rate, f1_score)
    return precision_rate, recall_rate, f1_score


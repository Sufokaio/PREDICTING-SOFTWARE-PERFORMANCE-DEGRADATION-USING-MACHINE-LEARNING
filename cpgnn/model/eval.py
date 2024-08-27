from typing import Tuple
import numpy as np
import tensorflow as tf
from sklearn import metrics

from util.setting import log

from model.load_gnn import GNNLoader
from model.SGL import SGL

def calculate_metrics(tp:int, fn:int, tn:int, fp:int) -> Tuple[float, float, float]:
    """"""
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1

def calculate_tpr(tp:int, fn:int) -> float:
    tpr = tp / (tp + fn)
    return tpr

def print_metrics(msg:str, rel:dict) -> None:
    """"""
    if rel['valid']:
        log.info('%s: [tp, fn, tn, fp]==[%d, %d, %d, %d] [rec, pre, f1, auc]==[%f, %f, %f, %f]'
                % (msg, 
                rel['tp'], rel['fn'], rel['tn'], rel['fp'], 
                rel['recall'], rel['precision'], rel['f1'], rel['auc']))

def print_tpr(msg:str, rel:dict) -> None:
    """"""
    if rel['valid']:
        log.info('%s: [tp, fn]==[%d, %d] [tpr]==[%f]'
                % (msg, rel['tp'], rel['fn'], rel['tpr']))

def classification_validation(sess:tf.Session, model:SGL, data_generator:GNNLoader, best_classification_validation:list) -> dict:
    """"""
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'accuracy':0.}

    if data_generator.n_classification_val == 0:
        rel['valid'] = 0
        log.warn("n_classification_val == 0")
        return rel

    n_batch_classification_val = data_generator.n_classification_val // data_generator.batch_size_classification
    if n_batch_classification_val == 0:
        n_batch_classification_val = 1
    elif data_generator.n_classification_val % data_generator.batch_size_classification:
        n_batch_classification_val += 1
    
    classification_rel = []
    classification_label = []

    for i_batch in range(n_batch_classification_val):
        batch_data = data_generator.generate_classification_val_batch(i_batch, (i_batch == n_batch_classification_val - 1))
        feed_dict = data_generator.generate_classification_val_feed_dict(model, batch_data)
        
        classification_rel.extend(model.eval_classification(sess, feed_dict))
        classification_label.extend(batch_data['y_classification'])

    classification_pred = np.argmax(classification_rel, axis=1)    
    rel['accuracy'] = metrics.accuracy_score(classification_label, classification_pred)

    if rel['accuracy'] > best_classification_validation[0]:
        best_classification_validation[0] = rel['accuracy']
    
    log.info('Classification Validation: [acc, best]==[%f, %f]' % (rel['accuracy'], best_classification_validation[0]))

def classification_test(sess:tf.Session, model:SGL, data_generator:GNNLoader) -> None:
    """"""
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'accuracy': 0.}

    if data_generator.n_classification_test == 0:
        rel['valid'] = 0
        log.warn("n_classification_test == 0")
        return rel

    n_batch_classification_test = data_generator.n_classification_test // data_generator.batch_size_classification
    if n_batch_classification_test == 0:
        n_batch_classification_test = 1
    elif data_generator.n_classification_test % data_generator.batch_size_classification:
        n_batch_classification_test += 1
    
    classification_rel = []
    classification_label = []

    for i_batch in range(n_batch_classification_test):
        batch_data = data_generator.generate_classification_test_batch(i_batch, (i_batch == n_batch_classification_test - 1))
        feed_dict = data_generator.generate_classification_val_feed_dict(model, batch_data)
        
        classification_rel.extend(model.eval_classification(sess, feed_dict))
        classification_label.extend(batch_data['y_classification'])

    classification_pred = np.argmax(classification_rel, axis=1)    
    rel['precision'], rel['recall'], rel['f1'], _ = metrics.precision_recall_fscore_support(classification_label, classification_pred, average='macro')
    rel['accuracy'] = metrics.accuracy_score(classification_label, classification_pred)

    log.info('Classification Test: [rec, pre, f1, acc]==[%f, %f, %f, %f]'
        % (rel['recall'], rel['precision'], rel['f1'], rel['accuracy']))

def perf_val_supervised(sess:tf.Session, model:SGL, data_generator:GNNLoader, threshold: float) -> None:
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'tpr': 0., 'auc': 0.}

    n_batch_perf = data_generator.n_perf_val // data_generator.batch_size_perf
    if data_generator.n_perf_val % data_generator.batch_size_perf:
        n_batch_perf += 1
    
    perf_rel = []
    perf_label = []

    for i_batch in range(n_batch_perf):
        batch_data = data_generator.generate_perf_val_batch(i_batch, (i_batch == n_batch_perf - 1))
        feed_dict = data_generator.generate_perf_feed_dict(model, batch_data)

        perf_rel.extend(model.eval_perf_supervised(sess, feed_dict))
        perf_label.extend(batch_data['y_perf'])

    perf_pred = np.array(perf_rel) > threshold

    rel['recall'] = metrics.recall_score(perf_label, perf_pred, average='binary')
    rel['precision'] = metrics.precision_score(perf_label, perf_pred, average='binary')
    rel['f1'] = metrics.f1_score(perf_label, perf_pred, average='binary')

    # note: input perf_rel rather than perf_pred
    fpr, tpr, thresholds = metrics.roc_curve(perf_label, perf_rel, pos_label=1)
    rel['auc'] = metrics.auc(fpr, tpr)

    log.info('perf Validation: [rec, pre, f1, auc]==[%f, %f, %f, %f]'
        % (rel['recall'], rel['precision'], rel['f1'], rel['auc']))

def perf_test_supervised(sess:tf.Session, model:SGL, data_generator:GNNLoader, threshold: float) -> None:
    rel = {'valid': 1, 'recall': 0., 'precision': 0., 'f1': 0., 'tpr': 0., 'auc': 0.}

    n_batch_perf = data_generator.n_perf_test_supervised // data_generator.batch_size_perf
    if data_generator.n_perf_test_supervised % data_generator.batch_size_perf:
        n_batch_perf += 1
    
    perf_rel = []
    perf_label = []
    for i_batch in range(n_batch_perf):
        batch_data = data_generator.generate_perf_test_batch(i_batch, (i_batch == n_batch_perf - 1))
        feed_dict = data_generator.generate_perf_feed_dict(model, batch_data)
        perf_rel.extend(model.eval_perf_supervised(sess, feed_dict))
        perf_label.extend(batch_data['y_perf'])

    perf_pred = np.array(perf_rel) > threshold

    rel['recall'] = metrics.recall_score(perf_label, perf_pred, average='binary')
    rel['precision'] = metrics.precision_score(perf_label, perf_pred, average='binary')
    rel['f1'] = metrics.f1_score(perf_label, perf_pred, average='binary')

    # note: input perf_rel rather than perf_pred
    fpr, tpr, thresholds = metrics.roc_curve(perf_label, perf_rel, pos_label=1)
    rel['auc'] = metrics.auc(fpr, tpr)

    log.info('perf Test: [rec, pre, f1, auc]==[%f, %f, %f, %f]'
        % (rel['recall'], rel['precision'], rel['f1'], rel['auc']))

def perf_test_unsupervised(sess:tf.Session, model:SGL, data_generator:GNNLoader, threshold:float) -> None:
    perf_threshold = threshold

    rel = {'tp': 0 , 'fn': 0, 'tn': 0, 'fp': 0, 'valid': 1, 
    'recall': 0., 'precision': 0., 'f1': 0., 'tpr': 0., 'auc': 0.}

    n_batch_perf = data_generator.n_perf_test_unsupervised // data_generator.batch_size_perf
    if data_generator.n_perf_test_unsupervised % data_generator.batch_size_perf:
        n_batch_perf += 1
    
    pos_rel = []
    neg_rel = []

    for i_batch in range(n_batch_perf):
        batch_data_pos = data_generator.generate_perf_batch(i_batch, (i_batch == n_batch_perf - 1), pos=True)
        feed_pos = data_generator.generate_perf_feed_dict(model, batch_data_pos)
        
        batch_data_neg = data_generator.generate_perf_batch(i_batch, (i_batch == n_batch_perf - 1), pos=False)
        feed_neg = data_generator.generate_perf_feed_dict(model, batch_data_neg)

        pos_rel.extend(model.eval_perf_unsupervised(sess, feed_pos))
        neg_rel.extend(model.eval_perf_unsupervised(sess, feed_neg))

    pos_pred = np.array(pos_rel) >= perf_threshold
    tp = np.sum(pos_pred)
    fn = pos_pred.shape[0] - tp
    rel['tp'] += tp
    rel['fn'] += fn

    neg_pred = np.array(neg_rel) >= perf_threshold
    fp = np.sum(neg_pred)
    tn = neg_pred.shape[0] - fp
    rel['fp'] += fp
    rel['tn'] += tn

    y = [1] * len(pos_rel) + [-1] * len(neg_rel)
    scores = pos_rel + neg_rel
    fpr, tpr, _ = metrics.roc_curve(y, scores)
    rel['auc'] = metrics.auc(fpr, tpr)

    rel['recall'], rel['precision'], rel['f1'] = calculate_metrics(rel['tp'], rel['fn'], rel['tn'], rel['fp'])
    
    print_metrics("perf Test Unsupervised", rel)

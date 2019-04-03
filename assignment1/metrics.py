import numpy as np
from sklearn.metrics import confusion_matrix

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    #Accuracy
    accuracy = np.sum(prediction == ground_truth) / len(prediction)
    
    #Precision and Recall
    tp = confusion_matrix(prediction, ground_truth)[0, 0]
    fp = confusion_matrix(prediction, ground_truth)[0, 1]
    tn = confusion_matrix(prediction, ground_truth)[1, 0]
    fn = confusion_matrix(prediction, ground_truth)[1, 1]
    precision = tp / (tp + fp)
    recall = tp / (tp + tn)
    
    
    f1 = 2 * precision*recall / (precision+recall)
    
    

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return (prediction == ground_truth).sum() / len(prediction)


from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.metrics import top_k_accuracy_score


class Metrics:
    def __init__(self, predictions, labels, logits=None):
        """
        Initialize the metric class with model predictions and ground truth labels.
        
        Args:
            predictions (np.ndarray or list): Predicted labels or top-k predictions.
            labels (np.ndarray or list): Ground truth labels.
            logits (np.ndarray, optional): Raw output from the model (before softmax), required for some metrics.
        """
        self.predictions = predictions
        self.labels = labels
        self.logits = logits
        self.results = {}
    
    def run_metric(self, metric_function, name=None):
        """
        Run a single metric function and store the result.
        
        Args:
            metric_function (function): A function that accepts (predictions, labels, logits).
            name (str, optional): Name under which to store the result.
                                  If not provided, use the function's __name__.
        """
        metric_name = name or metric_function.__name__
        self.results[metric_name] = metric_function(self.predictions, self.labels, self.logits)

    def get_results(self):
        """
        Get the computed metrics.
        
        Returns:
            dict: Dictionary of computed metric results.
        """
        return self.results


def f1_score_macro(preds, labels, logits=None):
    """
    Compute the macro-averaged F1 score.

    Args:
        preds (list or np.ndarray): Predicted labels.
        labels (list or np.ndarray): Ground truth labels.
        logits (optional): Not used for F1.

    Returns:
        float: Macro F1 score.
    """
    return f1_score(labels, preds, average='macro')



def mean_average_precision(preds, labels, logits=None):
    """
    Compute mean Average Precision (mAP) in a multi-class setting.

    Args:
        preds (list or np.ndarray): Predicted labels (not used here).
        labels (list or np.ndarray): Ground truth labels (int).
        logits (np.ndarray): Raw scores or predicted probabilities (n_samples x n_classes).

    Returns:
        float: mean Average Precision (macro)
    """
    if logits is None:
        raise ValueError("logits are required for computing mAP")

    labels = np.array(labels)
    num_classes = logits.shape[1]
    # Convert to one-hot
    y_true = label_binarize(labels, classes=np.arange(num_classes))

    return average_precision_score(y_true, logits, average="macro")


def top_k_accuracy(preds, labels, logits=None, k=5):
    """
    Compute Top-k Accuracy for multi-class classification.

    Args:
        preds (not used): Placeholder for compatibility.
        labels (list or np.ndarray): Ground truth labels.
        logits (np.ndarray): Model output scores (n_samples x n_classes).
        k (int): The "k" in Top-k.

    Returns:
        float: Top-k accuracy score.
    """
    if logits is None:
        raise ValueError("logits are required for top-k accuracy")

    return top_k_accuracy_score(labels, logits, k=k, labels=np.arange(logits.shape[1]))


def cmc_score(preds, labels, logits=None, topk=5):
    """
    Compute Cumulative Matching Characteristic (CMC) up to top-k ranks.

    Args:
        preds (not used): Placeholder.
        labels (list or np.ndarray): Ground truth labels.
        logits (np.ndarray): Similarity scores or confidence scores (n_samples x n_classes).
        topk (int): Maximum rank to compute.

    Returns:
        dict: {rank_k: accuracy_at_k} for k in 1..topk
    """
    if logits is None:
        raise ValueError("logits (as similarity/confidence scores) required for CMC")

    labels = np.array(labels)
    topk = min(topk, logits.shape[1])
    cmc = np.zeros(topk)

    for i, true_label in enumerate(labels):
        # Get ranking indices from highest to lowest score
        ranked_indices = np.argsort(logits[i])[::-1]
        # Find where the true label is ranked
        rank = np.where(ranked_indices == true_label)[0][0]
        if rank < topk:
            cmc[rank:] += 1

    cmc /= len(labels)
    return {f"CMC@{k+1}": round(cmc[k], 4) for k in range(topk)}

def mean_reciprocal_rank(preds, labels, logits=None):
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        preds (not used): Placeholder.
        labels (list or np.ndarray): Ground truth labels.
        logits (np.ndarray): Confidence/similarity scores (n_samples x n_classes).

    Returns:
        float: Mean Reciprocal Rank score.
    """
    if logits is None:
        raise ValueError("logits (as similarity/confidence scores) required for MRR")

    labels = np.array(labels)
    reciprocal_ranks = []

    for i, true_label in enumerate(labels):
        ranked_indices = np.argsort(logits[i])[::-1]
        rank = np.where(ranked_indices == true_label)[0][0]
        reciprocal_ranks.append(1.0 / (rank + 1))  # 1-based rank

    return round(np.mean(reciprocal_ranks), 4)


if __name__ == "__main__":
    # Dummy logits: shape (6 samples x 3 classes)
    logits = np.array([
        [0.9, 0.05, 0.05],  # True: 0
        [0.2, 0.7, 0.1],    # True: 1
        [0.1, 0.8, 0.1],    # True: 1
        [0.6, 0.2, 0.2],    # True: 0
        [0.3, 0.2, 0.5],    # True: 2
        [0.2, 0.1, 0.7]     # True: 2
    ])
    
    # Ground truth labels
    labels = [0, 1, 1, 0, 2, 2]
    
    # Predicted labels from logits
    preds = np.argmax(logits, axis=1)

    # Initialize the metric class
    metric = Metrics(predictions=preds, labels=labels, logits=logits)

    # Run each metric
    metric.run_metric(f1_score_macro)
    metric.run_metric(mean_average_precision)
    metric.run_metric(lambda p, l, lo: top_k_accuracy(p, l, lo, k=1), name="top_1_accuracy")
    metric.run_metric(lambda p, l, lo: top_k_accuracy(p, l, lo, k=3), name="top_3_accuracy")
    metric.run_metric(lambda p, l, lo: cmc_score(p, l, lo, topk=3), name="cmc")
    metric.run_metric(mean_reciprocal_rank)

    # Print the results
    import pprint
    pprint.pprint(metric.get_results())

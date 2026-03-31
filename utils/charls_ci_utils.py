import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, brier_score_loss, average_precision_score


def _cluster_uid_row_indices(groups):
    """Map each cluster ID (e.g. participant ID) to row indices."""
    groups = np.asarray(groups)
    by_uid = defaultdict(list)
    for i, g in enumerate(groups):
        by_uid[g].append(i)
    uids = np.unique(groups)
    uid_to_rows = {uid: np.array(by_uid[uid], dtype=int) for uid in uids}
    return uids, uid_to_rows


def cluster_bootstrap_indices_once(groups, rng, n_clusters_draw=None):
    """
    One cluster-bootstrap replicate: draw clusters (patient IDs) with replacement,
    then concatenate all person-waves belonging to each drawn cluster.
    If a cluster is drawn k times, its rows appear k times (standard clustered bootstrap).

    Parameters
    ----------
    groups : array-like of shape (n_rows,)
        Cluster label per row (e.g. CHARLS participant ID).
    rng : np.random.RandomState
    n_clusters_draw : int, optional
        Number of cluster draws; default len(unique clusters).

    Returns
    -------
    np.ndarray
        Row indices into the original table (length varies across replicates).
    """
    uids, uid_to_rows = _cluster_uid_row_indices(groups)
    n_c = len(uids)
    if n_c == 0:
        return np.array([], dtype=int)
    k = int(n_clusters_draw) if n_clusters_draw is not None else n_c
    sampled_uids = rng.choice(uids, size=k, replace=True)
    parts = [uid_to_rows[u] for u in sampled_uids]
    return np.concatenate(parts) if parts else np.array([], dtype=int)


def get_metrics_with_ci(y_true, y_prob, groups=None, n_bootstraps=1000, random_state=None):
    """
    计算指标及 95% Bootstrap 置信区间。

    - 若提供 ``groups``（与 y 等长的患者/聚类 ID）：使用 **cluster bootstrap**（对 ID 有放回抽样，
      再取该 ID 下全部 person-wave），适用于纵向重复测量，避免行级 bootstrap 低估方差。
    - 若 ``groups is None``：回退为 **分层行级 bootstrap**（仅适用于近似独立同分布行；遗留兼容）。

    **重要**：本函数中的 ``Recall`` / ``Recall_raw`` 基于固定概率阈值 **0.5**，在极度不平衡时
    会严重偏离临床可用阈值；**禁止**用于 CPM 冠军筛选或模型淘汰。冠军门槛应使用
    ``compare_models`` 写入的 ``Recall_at_opt_t_raw``（OOF Youden 最优阈值在测试集上的 Recall）。
    """
    from sklearn.metrics import confusion_matrix
    if random_state is None:
        try:
            from config import RANDOM_SEED
            random_state = RANDOM_SEED
        except ImportError:
            random_state = 500
    rng = np.random.RandomState(random_state)
    y_pred = (y_prob > 0.5).astype(int)

    def calculate_youden(y_t, y_p):
        pred = (y_p > 0.5).astype(int)
        cm = confusion_matrix(y_t, pred)
        if cm.shape != (2, 2) or len(np.unique(y_t)) < 2 or len(np.unique(pred)) < 2:
            return 0.0
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        return sensitivity + specificity - 1

    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = 0.5

    metrics = {
        'AUC': auc_val,
        'AUPRC': average_precision_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Youden': calculate_youden(y_true, y_prob),
        'Brier': brier_score_loss(y_true, y_prob)
    }

    bootstrapped_stats = {k: [] for k in metrics.keys()}
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y_true_arr):
            raise ValueError(f"groups length {len(groups)} != y_true length {len(y_true_arr)}")
        uids, _ = _cluster_uid_row_indices(groups)
        n_clust = len(uids)
        for _ in range(n_bootstraps):
            idx = cluster_bootstrap_indices_once(groups, rng, n_clusters_draw=n_clust)
            if len(idx) == 0:
                continue
            y_true_b = y_true_arr[idx]
            if len(np.unique(y_true_b)) < 2:
                continue
            y_prob_b = y_prob_arr[idx]
            y_pred_b = (y_prob_b > 0.5).astype(int)
            bootstrapped_stats['AUC'].append(roc_auc_score(y_true_b, y_prob_b))
            bootstrapped_stats['AUPRC'].append(average_precision_score(y_true_b, y_prob_b))
            bootstrapped_stats['Accuracy'].append(accuracy_score(y_true_b, y_pred_b))
            bootstrapped_stats['F1'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
            bootstrapped_stats['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
            bootstrapped_stats['Recall'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
            bootstrapped_stats['Youden'].append(calculate_youden(y_true_b, y_prob_b))
            bootstrapped_stats['Brier'].append(brier_score_loss(y_true_b, y_prob_b))
    else:
        idx_pos = np.where(y_true_arr == 1)[0]
        idx_neg = np.where(y_true_arr == 0)[0]
        n_pos, n_neg = len(idx_pos), len(idx_neg)
        for _ in range(n_bootstraps):
            if n_pos > 0 and n_neg > 0:
                i_pos = rng.choice(idx_pos, size=n_pos, replace=True)
                i_neg = rng.choice(idx_neg, size=n_neg, replace=True)
                indices = np.concatenate([i_pos, i_neg])
                rng.shuffle(indices)
            else:
                indices = rng.randint(0, len(y_true_arr), len(y_true_arr))
            y_true_b = y_true_arr[indices]
            if len(np.unique(y_true_b)) < 2:
                continue
            y_prob_b = y_prob_arr[indices]
            y_pred_b = (y_prob_b > 0.5).astype(int)
            bootstrapped_stats['AUC'].append(roc_auc_score(y_true_b, y_prob_b))
            bootstrapped_stats['AUPRC'].append(average_precision_score(y_true_b, y_prob_b))
            bootstrapped_stats['Accuracy'].append(accuracy_score(y_true_b, y_pred_b))
            bootstrapped_stats['F1'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
            bootstrapped_stats['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
            bootstrapped_stats['Recall'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
            bootstrapped_stats['Youden'].append(calculate_youden(y_true_b, y_prob_b))
            bootstrapped_stats['Brier'].append(brier_score_loss(y_true_b, y_prob_b))

    final_results = {}
    for k, v in metrics.items():
        arr = bootstrapped_stats[k]
        if len(arr) < 10:
            low, high = v, v
        else:
            low = np.percentile(arr, 2.5)
            high = np.percentile(arr, 97.5)
        final_results[k] = f"{v:.4f} (95% CI: {low:.4f}, {high:.4f})"
        final_results[f'{k}_raw'] = v
    return final_results

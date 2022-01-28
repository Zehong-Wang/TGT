import torch
import math
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import numpy as np

def recall(node, rank, top_k):
    rank = rank[:, :top_k]
    recall = np.array([node[i] in a for i, a in enumerate(rank)])
    recall = recall.sum() / recall.size
    return recall

def MRR(node, rank):
    rank = rank.cpu()
    mrr = np.array([(np.where(a==node[i])) for i, a in enumerate(rank)])
    mrr = (1 / (mrr + 1)).mean()
    return mrr

def get_target(src_embedding, dest_embedding, src_batch):
    cos_similarity = torch.matmul(src_embedding[src_batch], dest_embedding.T)
    cos_similarity, idx = torch.sort(cos_similarity, descending=True)
    return cos_similarity, idx

def eval_edge_prediction(model, neg_edge_sampler, data, n_neighbors, batch_size=200):
    assert neg_edge_sampler.seed is not None
    neg_edge_sampler.reset_random_state()

    val_ap, val_macro_auc, val_micro_auc, val_macro_f1, val_micro_f1 = [], [], [], [], []
    val_mrr, val_recall_20, val_recall_50 = [], [], []

    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            start_idx = k * TEST_BATCH_SIZE
            end_idx = min(num_test_instance, start_idx + TEST_BATCH_SIZE)
            size = end_idx - start_idx

            src_batch = data.sources[start_idx:end_idx]
            dest_batch = data.destinations[start_idx:end_idx]
            edge_idx_batch = data.edge_idxs[start_idx:end_idx]
            timestamp_batch = data.timestamps[start_idx:end_idx]
            _, neg_batch = neg_edge_sampler.sample(size)

            pos_prob, neg_prob = model(src_node=src_batch,
                                       dest_node=dest_batch,
                                       neg_node=neg_batch,
                                       edge_time=timestamp_batch,
                                       edge_src_dest_idx=edge_idx_batch,
                                       n_neighbors=n_neighbors)

            pred_label = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_label))
            val_macro_auc.append(roc_auc_score(true_label, pred_label, average='macro'))
            val_micro_auc.append(roc_auc_score(true_label, pred_label, average='micro'))
            val_macro_f1.append(f1_score(true_label, np.array(pred_label>=0.5, dtype=int), average='macro'))
            val_micro_f1.append(f1_score(true_label, np.array(pred_label>=0.5, dtype=int), average='micro'))

            src_embedding = model.embedding
            dest_embedding = model.embedding
            cos_similarity, dest_rank = get_target(src_embedding, dest_embedding, src_batch)
            cos_similarity, src_rank = get_target(dest_embedding, src_embedding, dest_batch)

            recall_20 = (recall(dest_batch, dest_rank, 20) + recall(src_batch, src_rank, 20)) / 2
            recall_50 = (recall(dest_batch, dest_rank, 50) + recall(src_batch, src_rank, 50)) / 2
            mrr = (MRR(dest_batch, dest_rank) + MRR(src_batch, src_rank)) / 2

            val_mrr.append(mrr)
            val_recall_20.append(recall_20)
            val_recall_50.append(recall_50)

    return np.mean(val_ap), np.mean(val_macro_auc), np.mean(val_micro_auc), np.mean(val_macro_f1), np.mean(val_micro_f1), np.mean(val_mrr), np.mean(val_recall_20), np.mean(val_recall_50)



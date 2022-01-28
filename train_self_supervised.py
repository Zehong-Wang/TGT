import argparse
import math
import pickle
import sys
import time
import numpy as np
import torch
from evaluate.evaluation import eval_edge_prediction
from tqdm import tqdm
from utils.utils import RandEdgeSampler, get_neighbor_finder, get_data, compute_time_statistics, EarlyStopMonitor
from model.tgt import TGT


parser = argparse.ArgumentParser('TGT self-supervised training')
parser.add_argument('--data', type=str, default='uci', help='Dataset name')
parser.add_argument('--bs', type=int, default=20, help='Batch size')
parser.add_argument('--epoch_num', type=int, default=50, help='Epoch number')
parser.add_argument('--neighbor_num', type=int, default=20, help='Number of neighbors')
parser.add_argument('--aggregator', type=str, default='mean', help='Aggregator type')
parser.add_argument('--memory_dim', type=int, default=256, help='Memory dimension')
parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension')
parser.add_argument('--time_dim', type=int, default=256, help='Time embedding dimension')
parser.add_argument('--transformer_layer', type=int, default=1, help='Transformer layer number')
parser.add_argument('--attn_head', type=int, default=2, help='Attention head number')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

dataset_name = args.data
EPOCH_NUM = args.epoch_num
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.neighbor_num
AGGREGATOR = args.aggregator
MEMORY_DIM = args.memory_dim
EMBEDDING_DIM = args.embedding_dim
TIME_DIM = args.time_dim
TRANSFORMER_LAYER = args.transformer_layer
ATTN_HEAD = args.attn_head

get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{dataset_name}-{epoch}.pth'

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(
    dataset_name, different_new_nodes_between_val_and_test=True, randomize_features=False)

train_ngh_finder = get_neighbor_finder(train_data, 'recent')
full_ngh_finder = get_neighbor_finder(full_data, 'recent')

train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

device_string = 'cuda:{}'.format('0') if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

results_path = 'results/tgt-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(dataset_name, BATCH_SIZE, AGGREGATOR, MEMORY_DIM, EMBEDDING_DIM, TIME_DIM, TRANSFORMER_LAYER, ATTN_HEAD)

mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(
    full_data.sources, full_data.destinations, full_data.timestamps)

tgt = TGT(neighbor_finder=train_ngh_finder,
          d_node_memory=MEMORY_DIM,
          d_node_embedding=EMBEDDING_DIM,
          d_time_embedding=TIME_DIM,
          n_layers=TRANSFORMER_LAYER,
          n_head=ATTN_HEAD,
          d_k=256,
          d_v=256,
          d_inner=256,
          edge_features=edge_features,
          aggregator=AGGREGATOR,
          src_time_shift_mean=mean_time_shift_src,
          src_time_shift_std=std_time_shift_src,
          dest_time_shift_mean=mean_time_shift_dst,
          dest_time_shift_std=std_time_shift_dst,
          device=device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(tgt.parameters(), lr=0.0001)
tgt = tgt.to(device)

early_stopper = EarlyStopMonitor(max_round=5)

new_node_val_aps = list()
val_aps = list()
epoch_times = list()
total_epoch_times = list()
train_losses = list()

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

idx_list = np.arange(num_instance)

for epoch in range(EPOCH_NUM):
    start_epoch = time.time()

    tgt.set_neighbor_finder(train_ngh_finder)

    m_loss = list()

    for k in tqdm(range(num_batch)):
        tgt = tgt.train()
        loss = 0
        optimizer.zero_grad()

        batch_idx = k
        if batch_idx >= num_batch:
            continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        size = end_idx - start_idx

        src_batch = train_data.sources[start_idx:end_idx]
        dest_batch = train_data.destinations[start_idx:end_idx]
        edge_idx_batch = train_data.edge_idxs[start_idx:end_idx]
        timestamp_batch = train_data.timestamps[start_idx:end_idx]
        _, neg_batch = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        pos_prob, neg_prob = tgt(src_node=src_batch,
                                 dest_node=dest_batch,
                                 neg_node=neg_batch,
                                 edge_time=timestamp_batch,
                                 edge_src_dest_idx=edge_idx_batch)
        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
        loss.backward()
        optimizer.step()
        m_loss.append(loss.item())

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    # Validation
    tgt.set_neighbor_finder(full_ngh_finder)

    memory_backup = tgt.memory.clone()
    embedding_backup = tgt.embedding.clone()

    val_ap, val_macro_auc, val_micro_auc, val_macro_f1, val_micro_f1, val_mrr, val_recall_20, val_recall_50 = eval_edge_prediction(model=tgt,
                                           neg_edge_sampler=val_rand_sampler,
                                           data=val_data,
                                           n_neighbors=NUM_NEIGHBORS,
                                           batch_size=BATCH_SIZE)

    # restore memory and embedding
    val_memory_backup = tgt.memory.clone()
    val_embedding_backup = tgt.embedding.clone()
    tgt.memory = memory_backup
    tgt.embedding = embedding_backup

    nn_val_ap, nn_val_macro_auc, nn_val_micro_auc, nn_val_macro_f1, nn_val_micro_f1, nn_val_mrr, nn_val_recall_20, nn_val_recall_50 = eval_edge_prediction(model=tgt,
                                                 neg_edge_sampler=nn_val_rand_sampler,
                                                 data=new_node_val_data,
                                                 n_neighbors=NUM_NEIGHBORS,
                                                 batch_size=BATCH_SIZE)

    # restore validation memory and embedding
    tgt.memory = val_memory_backup
    tgt.embedding = val_embedding_backup

    new_node_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    pickle.dump({
        'val_aps': val_aps,
        'new_node_val_aps': new_node_val_aps,
        'val_macro_auc': val_macro_auc,
        'new_node_val_macro_auc': nn_val_macro_auc,
        'val_micro_auc': val_micro_auc,
        'new_node_val_micro_auc': nn_val_micro_auc,
        'val_macro_f1': val_macro_f1,
        'new_node_val_macro_f1': nn_val_macro_f1,
        'val_micro_f1': val_micro_f1,
        'new_node_val_micro_f1': nn_val_micro_f1,
        'val_mrr': val_mrr,
        'new_node_val_mrr': nn_val_mrr,
        'val_recall_20': val_recall_20,
        'new_node_val_recall_20': nn_val_recall_20,
        'val_recall_50': val_recall_50,
        'new_node_val_recall_50': nn_val_recall_50,
        'epoch_times': epoch_times,
        'train_losses': train_losses,
        'total_epoch_times': total_epoch_times
    }, open(results_path, 'wb'))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    print('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    print('Epoch mean loss: {}'.format(np.mean(m_loss)))
    print('val macro auc: {}, new node val macro auc: {}'.format(val_macro_auc, nn_val_macro_auc))
    print('val micro auc: {}, new node val micro auc: {}'.format(val_micro_auc, nn_val_micro_auc))
    print('val macro f1: {}, new node val macro f1: {}'.format(val_macro_f1, nn_val_macro_f1))
    print('val micro f1: {}, new node val micro f1: {}'.format(val_micro_f1, nn_val_micro_f1))
    print('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    print('val mrr: {}, new node val mrr: {}'.format(val_mrr, nn_val_mrr))
    print('val recall 20: {}, new node val recall 20: {}'.format(val_recall_20, nn_val_recall_20))
    print('val recall 50: {}, new node val recall 50: {}'.format(val_recall_50, nn_val_recall_50))

    if early_stopper.early_stop_check(val_macro_auc):
        print('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
        print('Loading the best model at epoch {}'.format(early_stopper.best_epoch))
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        tgt.load_state_dict(torch.load(best_model_path))
        tgt = tgt.eval()
        break
    else:
        torch.save(tgt.state_dict(), get_checkpoint_path(epoch))

val_memory_backup = tgt.memory.clone()
val_embedding_backup = tgt.embedding.clone()

# Test
test_ap, test_macro_auc, test_micro_auc, test_macro_f1, test_micro_f1, test_mrr, test_recall_20, test_recall_50 = eval_edge_prediction(model=tgt,
                                         neg_edge_sampler=test_rand_sampler,
                                         data=test_data,
                                         n_neighbors=NUM_NEIGHBORS,
                                         batch_size=BATCH_SIZE)

tgt.memory = val_memory_backup
tgt.embedding = val_embedding_backup

nn_test_ap, nn_test_macro_auc, nn_test_micro_auc, nn_test_macro_f1, nn_test_micro_f1, nn_test_mrr, nn_test_recall_20, nn_test_recall_50 = eval_edge_prediction(model=tgt,
                                               neg_edge_sampler=nn_test_rand_sampler,
                                               data=new_node_test_data,
                                               n_neighbors=NUM_NEIGHBORS,
                                               batch_size=BATCH_SIZE)


print('Test statistics: Old nodes -- macro auc: {}, micro auc: {}, macro f1: {}, micro f1: {}, ap: {}, mrr: {}, recall_20: {}, recall_50: {}'.format(test_macro_auc, test_micro_auc, test_macro_f1, test_micro_f1, test_ap, test_mrr, test_recall_20, test_recall_50))
print('Test statistics: New nodes -- macro auc: {}, micro auc: {}, macro f1: {}, micro f1: {}, ap: {}, mrr: {}, recall_20: {}, recall_50: {}'.format(nn_test_macro_auc, nn_test_micro_auc, nn_test_macro_f1, nn_test_micro_f1, nn_test_ap, nn_test_mrr, nn_test_recall_20, nn_test_recall_50))


pickle.dump({
    'test_ap': test_ap,
    'new_node_test_ap': nn_test_ap,
    'test_macro_auc': test_macro_auc,
    'new_node_test_macro_auc': nn_test_macro_auc,
    'test_micro_auc': test_micro_auc,
    'new_node_test_micro_auc': nn_test_micro_auc,
    'test_macro_f1': test_macro_f1,
    'new_node_test_macro_f1': nn_test_macro_f1,
    'test_micro_f1': test_micro_f1,
    'new_node_test_micro_f1': nn_test_micro_f1,
    'test_mrr': test_mrr,
    'new_node_test_mrr': nn_test_mrr,
    'test_recall_20': test_recall_20,
    'new_node_test_recall_20': nn_test_recall_20,
    'test_recall_50': test_recall_50,
    'new_node_test_recall_50': nn_test_recall_50,
    'epoch_times': epoch_times,
    'train_losses': train_losses,
    'total_epoch_times': total_epoch_times
}, open(results_path, 'wb'))

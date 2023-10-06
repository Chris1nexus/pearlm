import os

import numpy as np
import torch


def compute_topks(cf_scores, train_user_dict, valid_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for u in user_ids:
        train_pos_item_list = train_user_dict[u]
        valid_pos_item_list = valid_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[u][train_pos_item_list] = -np.inf
        cf_scores[u][valid_pos_item_list] = -np.inf
        test_pos_item_binary[u][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    topk_items_dict = {}  # Dictionary to store top-k items for each user
    maxK = max(Ks)
    for u in user_ids:
        topk_items = [item_ids[i] for i in rank_indices[u]][:maxK]  # Convert indices to real item IDs
        topk_items_dict[u] = topk_items
    return topk_items_dict


def early_stopping(metric_list, stopping_steps):
    best_recall = max(metric_list)
    best_step = metric_list.index(best_recall)
    if len(metric_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

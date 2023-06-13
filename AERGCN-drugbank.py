import csv
import gc
import os
from math import ceil

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import numpy as np
import argparse
import time

from models import RGCN, GAT, GCN, SAGE, AERGCNDDI
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import sample_mask
import warnings
import utils2
from sklearn.preprocessing import label_binarize

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from datetime import datetime
from sklearn.decomposition import PCA
from torch_geometric.data import Data
import argparse, time, torch, logging, os
warnings.filterwarnings('ignore')
exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M')+'GraphSage-drugbank65'
log_file_path = './log/'+exec_name+'.log'
logger = utils2.setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='stance', help='detection task of stance or bot')
parser.add_argument('--relation_select', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64], nargs='+', help='selection of relations in the graph (0-6)')
parser.add_argument('--random_seed', type=int, default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64], nargs='+', help='selection of random seeds')
parser.add_argument('--model', type=str, default='AERGCN', help='GCN, GAT, GraphSage, RGCN, AERGCN')
parser.add_argument('--hidden_dimension', type=int, default=256, help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (1 - keep probability)')
parser.add_argument('--epochs', type=int, default=900, help='training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


def main(seed):

    log_path_auc = './log/drugbank65-all.csv'
    file_all = open(log_path_auc, 'a+', encoding='utf-8', newline='')
    names2ids = utils2.get_mappings_drugbank_name()
    names2ids_k, tru_labels_k, feature_embedding_k, edge_index_k, fingerprint2ids_knwon = utils2.get_mappings_drugbank_fingerprint_after_classes(
        names2ids, "./Dataset/drug65/5-fold/1_kown_data_ddi.csv")

    # 获取未知-已知药物DDI 作为测试集合
    names2ids_un_k, tru_labels_un_k, feature_embedding_un_k, edge_index_un_k ,fingerprint2ids_un_k= utils2.get_mappings_drugbank_fingerprint_after_classes(names2ids,"./Dataset/drug65/5-fold/1_kown_unknown_data_ddi.csv")
    # 获取未知-未知药物DDI 作为测试集合
    names2ids_un, tru_labels_un, feature_embedding_un, edge_index_un,fingerprint2ids_unknwon= utils2.get_mappings_drugbank_fingerprint_unknown(names2ids)

    assert args.task in ['stance', 'bot'], "args.task should be choose from ['stance', 'bot']"
    out_dim = 65
    sample_number = len(edge_index_k[1])

    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.8 * sample_number)]
    val_idx = shuffled_idx[int(0.8 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]
    train_mask = sample_mask(train_idx, sample_number)
    val_mask = sample_mask(val_idx, sample_number)
    test_mask = sample_mask(test_idx, sample_number)
    relation_num = 65
    embedding_size = 600
    edge_type = tru_labels_k
    index_select_list = (edge_type == 65)
    args.relation_select= np.arange(0, 65, 1)

    feature_embedding_k = torch.tensor(feature_embedding_k)
    edge_index_k = torch.tensor(edge_index_k)
    edge_type = torch.tensor(edge_type)

    feature_embedding_un_k = torch.tensor(feature_embedding_un_k)
    edge_index_un_k = torch.tensor(edge_index_un_k)
    tru_labels_un_k = torch.tensor(tru_labels_un_k)

    feature_embedding_un = torch.tensor(feature_embedding_un)
    edge_index_un = torch.tensor(edge_index_un)
    tru_labels_un = torch.tensor(tru_labels_un)


    if torch.cuda.is_available():
        feature_embedding_k = feature_embedding_k.cuda()
        edge_index_k = edge_index_k.cuda()
        edge_type = edge_type.cuda()

        feature_embedding_un_k = feature_embedding_un_k.cuda()
        edge_index_un_k = edge_index_un_k.cuda()
        tru_labels_un_k = tru_labels_un_k.cuda()

        feature_embedding_un = feature_embedding_un.cuda()
        edge_index_un = edge_index_un.cuda()
        tru_labels_un = tru_labels_un.cuda()


    if args.model == 'GCN':
        model = GCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GAT':
        model = GAT(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GraphSage':
        model = SAGE(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'AERGCN':
        model = AERGCNDDI(256, embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)

    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss = loss.cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    batchsize = 3000

    def train(epoch):
        model.train()
        out_all = []
        scores_all=[]
        loss_all = 0
        cores_all_val = []
        train_all_label = []
        out_all_label = []
        val__all_label = []
        val_out_all_label = []
        out_val = []
        val_all_label = []
        batch_total = int(sample_number/batchsize) + 1
        for i in range(batch_total):
            if batchsize*(i + 1) < sample_number:
                feature_embedding = feature_embedding_k[batchsize * i:batchsize*(i + 1)]
                edge_type_train = edge_type[batchsize * i:batchsize * (i + 1)]
                edge_index = edge_index_k.t()[:batchsize]
            else:
                feature_embedding = feature_embedding_k[batchsize * i:]
                edge_type_train = edge_type[batchsize * i:]
                edge_index = edge_index_k.t()[batchsize * i:]

            # 计算val
            sample_number_i = len(feature_embedding)
            shuffled_idx = shuffle(np.array(range(sample_number_i)), random_state=seed)
            train_idx = shuffled_idx[:int(0.8 * sample_number_i)]
            val_idx = shuffled_idx[int(0.8 * sample_number_i):int(1 * sample_number_i)]
            # test_idx = shuffled_idx[int(0.9 * sample_number_i):]
            train_mask = sample_mask(train_idx, sample_number_i)
            val_mask = sample_mask(val_idx, sample_number_i)

            output = model(fingerprint2ids_knwon,feature_embedding,edge_index.t(), edge_type_train)
            loss_train = loss(output[train_mask], torch.tensor( edge_type_train)[train_mask])
            loss_all+=loss_train
            # loss_train.to(device)
            out = output.max(1)[1].to('cpu').detach().numpy()
            out = np.argmax(output.cpu().detach().numpy(), axis=1)
            out_all.extend(out[train_mask])
            out_val.extend(out[val_mask])
            # label = edge_type.detach().numpy()
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            train_all_label += edge_type_train[train_mask]
            val_all_label += edge_type_train[val_mask]
            out_all_label.extend( out[train_mask])
            scores_all.extend( output[train_mask])
            val__all_label += edge_type_train[val_mask]
            val_out_all_label.extend(out[val_mask])
            cores_all_val.extend( output[val_mask])

        label = np.asarray(torch.tensor(train_all_label).cpu())
        # val true lable
        tru_label_val = np.asarray(torch.tensor(val_all_label).cpu())
        val_label = np.asarray(torch.tensor(val__all_label).cpu())
        acc_train = accuracy_score(out_all_label, label)
        acc_val = accuracy_score(val_out_all_label, val_label )
        optimizer.zero_grad()
        event_num = 65
        optimizer.step()

        y_one_hot = label_binarize(label, classes=np.arange(event_num))
        y_one_hot_val = label_binarize(val_label, classes=np.arange(event_num))
        pred_one_hot = label_binarize(out_all_label, classes=np.arange(event_num))

        # 测试集
        scores_all_digit =  torch.stack(scores_all).cpu().detach().numpy()
        roc_auc_test = roc_auc_score(y_one_hot,scores_all_digit, average='micro')
        precision_score_test = precision_score(label, out_all, average='micro')
        recall_score_test = recall_score(label, out_all, average='micro')

        # 验证集
        scores_val_digit = torch.stack(cores_all_val).cpu().detach().numpy()
        roc_auc_val = roc_auc_score(y_one_hot_val, scores_val_digit, average='micro')
        precision_score_val = precision_score(tru_label_val , out_val, average='micro')
        recall_score_val = recall_score(tru_label_val , out_val, average='micro')

        print('Epoch: {:04d}'.format(epoch + 1),
              'roc_auc_score: {:.4f}'.format(roc_auc_test.item()),
              'precision_score: {:.4f}'.format(precision_score_test.item()),
              'recall_score: {:.4f}'.format(recall_score_test.item()), )
        print("**********************************************************************************************")
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_all.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'acc_val: {:.4f}'.format(acc_val.item()), )

        print('Epoch: {:04d}'.format(epoch + 1),
              'roc_auc_score: {:.4f}'.format(roc_auc_val.item()),
              'precision_score: {:.4f}'.format(precision_score_val.item()),
              'recall_score: {:.4f}'.format(recall_score_val.item()), )
        print("**********************************************************************************************")

        gc.collect()
        torch.cuda.empty_cache()
        return acc_val

    #
    # def test():
    #     model.eval()
    #     output = model(feature_embedding_k, edge_index_k, edge_type)
    #     loss_test = loss(output[test_mask], torch.tensor(edge_type)[test_mask])
    #     # loss_test = loss(output[data.test_mask], edge_type[data.test_mask])
    #     out = output.max(1)[1].to('cpu').detach().numpy()
    #     # label = edge_type.detach().numpy()
    #     label = np.asarray(edge_type.cpu())
    #     acc_test = accuracy_score(out[test_mask], label[test_mask])
    #     f1 = f1_score(out[test_mask], label[test_mask], average='macro')
    #     precision = precision_score(out[test_mask], label[test_mask], average='macro')
    #     recall = recall_score(out[test_mask], label[test_mask], average='macro')
    #
    #     # 补充数据
    #     event_num = 65
    #     y_one_hot = label_binarize(label, classes=np.arange(event_num))
    #     roc_auc_test = roc_auc_score(y_one_hot[test_mask], output[test_mask].cpu().detach().numpy(), average='micro')
    #     precision_score_test = precision_score(label[test_mask], out[test_mask], average='micro')
    #     recall_score_test = recall_score(label[test_mask], out[test_mask], average='micro')
    #
    #     return acc_test, loss_test, f1, precision, recall , roc_auc_test,precision_score_test,recall_score_test

    def test4unknown_kown():
        model.eval()
        output = model(fingerprint2ids_un_k,feature_embedding_un_k, edge_index_un_k, tru_labels_un_k)
        loss_test = loss(output, torch.tensor(tru_labels_un_k))
        # loss_test = loss(output[data.test_mask], edge_type[data.test_mask])
        out = output.max(1)[1].to('cpu').detach().numpy()
        # label = edge_type.detach().numpy()
        label = np.asarray(tru_labels_un_k.cpu())
        acc_test = accuracy_score(out, label)
        f1 = f1_score(out, label, average='micro')
        precision = precision_score(out, label, average='macro')
        recall = recall_score(out, label, average='macro')

        # 补充数据
        event_num = 65
        y_one_hot = label_binarize(label, classes=np.arange(event_num))
        roc_auc_test = roc_auc_score(y_one_hot, output.cpu().detach().numpy(), average='micro')
        precision_score_test = precision_score(label, out, average='micro')
        recall_score_test = recall_score(label, out, average='micro')
        roc_aupr_score = utils2.roc_aupr_score(y_one_hot, output.cpu().detach().numpy(), average='macro')

        return acc_test, loss_test, f1, precision, recall , roc_auc_test,precision_score_test,recall_score_test,roc_aupr_score

    def test4unknown():
        model.eval()
        output = model(fingerprint2ids_unknwon,feature_embedding_un, edge_index_un, tru_labels_un)
        loss_test = loss(output, torch.tensor(tru_labels_un))
        # loss_test = loss(output[data.test_mask], edge_type[data.test_mask])
        # out = output.max(1)[1].to('cpu').detach().numpy()
        out = np.argmax(output.cpu().detach().numpy(), axis=1)
        # label = edge_type.detach().numpy()
        label = np.asarray(tru_labels_un.cpu())
        acc_test = accuracy_score(out, label)
        f1 = f1_score(out, label, average='macro')
        precision = precision_score(out, label, average='macro')
        recall = recall_score(out, label, average='macro')

        # 补充数据
        event_num = 65
        y_one_hot = label_binarize(label, classes=np.arange(event_num))
        roc_auc_test = roc_auc_score(y_one_hot, output.cpu().detach().numpy(), average='micro')
        precision_score_test = precision_score(label, out, average='micro')
        recall_score_test = recall_score(label, out, average='micro')
        roc_aupr_score = utils2.roc_aupr_score(y_one_hot, output.cpu().detach().numpy(), average='micro')

        gc.collect()
        torch.cuda.empty_cache()

        return acc_test, loss_test, f1, precision, recall , roc_auc_test,precision_score_test,recall_score_test,roc_aupr_score

    model.apply(init_weights)


    max_val_acc = 0
    for epoch in range(args.epochs):
        acc_val = train(epoch)
        # acc_test, loss_test, f1, precision, recall  , roc_auc_test,precision_score_test,recall_score_test,roc_aupr_score=test4unknown_kown()
        acc_test, loss_test, f1, precision, recall, roc_auc_test, precision_score_test, recall_score_test,roc_aupr_score = test4unknown()
        print("TestIIII")
        print("Test set results:",
              "epoch= {:}".format(epoch),
              "test_accuracy= {:.4f}".format(acc_test),
              "precision= {:.4f}".format(precision_score_test),
              "recall= {:.4f}".format(recall),
              "f1_score= {:.4f}".format(f1),

              "roc_auc_test= {:.4f}".format(roc_auc_test),
              "precision_score_test= {:.4f}".format(precision_score_test),
              "recall_score_test= {:.4f}".format(recall_score_test),
              "roc_aupr_score= {:.4f}".format(roc_aupr_score)
              )
        print("TestIIII")
        logger.info(
            f"-Test set results:-epoch= {epoch}-test_accuracy= {acc_test}-precision= {precision}-recall= {recall}-f1_score= {f1}-roc_auc_test= {roc_auc_test}-precision_score_test= {precision_score_test}-recall_score_test= {recall_score_test}-roc_aupr_score= {roc_aupr_score}")

        # 记录AUC到csv  AUC AUPR ACC  F-score
        csv_writer_all = csv.writer(file_all)
        csv_writer_all.writerow([epoch, '%.4f' %roc_auc_test,'%.4f' %roc_aupr_score,'%.4f' %acc_test,'%.4f' %f1])



        if acc_test > max_val_acc:
            max_val_acc = acc_test
            max_acc = acc_test
            max_epoch = epoch + 1
            max_f1 = f1
            max_precision = precision
            max_recall = recall

            max_roc_auc_test = roc_auc_test
            max_precision_score_test = precision_score_test
            max_recall_score_test = recall_score_test
            # max_roc_aupr_score = roc_aupr_score

    print("TestIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    print("Test set results:",
          "epoch= {:}".format(max_epoch),
          "test_accuracy= {:.4f}".format(max_acc),
          "precision= {:.4f}".format(max_precision),
          "recall= {:.4f}".format(max_recall),
          "f1_score= {:.4f}".format(max_f1),

          "max_roc_auc_test= {:.4f}".format(max_roc_auc_test),
          "max_precision_score_test= {:.4f}".format(max_precision_score_test),
          "max_recall_score_test= {:.4f}".format(max_recall_score_test),
          # "max_roc_aupr_score= {:.4f}".format(max_roc_aupr_score)
          )
    print("TestIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
    return max_acc, max_precision, max_recall, max_f1 ,max_roc_auc_test,max_precision_score_test,max_recall_score_test


if __name__ == "__main__":

    t = time.time()
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []

    for i, seed in enumerate(args.random_seed):
        print('\ntraning {}th model'.format(i+1))
        acc, precision, recall, f1,max_roc_auc_test,max_precision_score_test,max_recall_score_test = main(seed)
        acc_list.append(acc*100)
        precision_list.append(precision*100)
        recall_list.append(recall*100)
        f1_list.append(f1*100)


    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
    print('total time:', time.time() - t)





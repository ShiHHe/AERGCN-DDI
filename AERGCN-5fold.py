import csv
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from sklearn.metrics import roc_auc_score
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
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from sklearn.model_selection import KFold,StratifiedKFold
from statistics import mean, stdev
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import label_binarize
from datetime import datetime
from utils2 import setup_logger
import argparse, time, torch, logging, os

exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M')+'GraphSage'
log_file_path = './log/'+exec_name+'.log'
logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)

warnings.filterwarnings('ignore')

torch.cuda.device_count()

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='stance', help='detection task of stance or bot')
parser.add_argument('--relation_select', type=int, default=[0,1,2,3], nargs='+', help='selection of relations in the graph (0-6)')
parser.add_argument('--random_seed', type=int, default=[0,1,2,3], nargs='+', help='selection of random seeds')
parser.add_argument('--model', type=str, default='AERGCN', help='GCN, GAT, GraphSage, RGCN, AERGCN')
parser.add_argument('--hidden_dimension', type=int, default=256, help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (1 - keep probability)')
parser.add_argument('--epochs', type=int, default=1000, help='training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for optimizer')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


def main(seed):

    x = utils2.prepare_features()
    # edge_index, edge_label = utils2.get_edge_label_attri("./Dataset/DDInter/5-fold/fold-5_kown_data_ddi.txt")
    edge_index, edge_label = utils2.get_edge_label_attri("Dataset/drugdata/DrugPairs-2.txt")
    x = torch.tensor(x)  # 转float
    # svd 降维
    # x = PCA(n_components=250).fit_transform(x).astype('float32')  # 降维到100维
    x = torch.tensor(x)
    x_feature = x
    # edge_index_emb = torch.Tensor(79560,200)
    edge_index_emb= []
    for emb in torch.tensor(edge_index).t():
        emb1 = emb[0].item()
        emb2 = emb[1].item()
        edge_index_emb_triple = torch.FloatTensor(torch.cat([x_feature[emb1], x_feature[emb2]], -1) )
        # edge_index_emb.add(edge_index_emb_triple)
        edge_index_emb.append(edge_index_emb_triple)
    # //////////////////////////////////////////

    edge_index_emb = torch.stack(edge_index_emb)
    datasetDDI = Data(x, edge_index=torch.tensor(edge_index), edge_type=torch.tensor(edge_label),edge_weight=torch.tensor(edge_label))

    data = datasetDDI
    sample_number = len(data.edge_index[1])
    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.7 * sample_number)]
    val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]
    data.train_mask = sample_mask(train_idx, sample_number)
    data.val_mask = sample_mask(val_idx, sample_number)
    data.test_mask = sample_mask(test_idx, sample_number)

    data = data.to(device)
    embedding_size = data.x.shape[1]
    relation_num = len(args.relation_select)
    index_select_list = (data.edge_type == 100)

    edge_index = data.edge_index[:, index_select_list]
    edge_type = data.edge_type[index_select_list]
    edge_weight =  data.edge_weight[index_select_list]

    out_dim = 5
    if args.model == 'RGCN':
        model = RGCN(embedding_size, args.hidden_dimension, out_dim, relation_num, args.dropout).to(device)
    elif args.model == 'GCN':
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

    edge_index_emb = edge_index_emb.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    batchsize =3000
    def train_l(idx,epoch,train_index_l):
        sample_number = len(train_index_l)
        model.train()
        pre_scores_all = []
        batch_total = int(sample_number/batchsize) + 1

        edge_index_emb_trian = edge_index_emb[train_index_l]
        edge_type_train = edge_type[train_index_l]
        edge_index_train = edge_index.t()[train_index_l]
        for i in range(batch_total-1):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            if batchsize*(i + 2) < sample_number:
                feature_embedding_train_t = edge_index_emb_trian[batchsize * i:batchsize*(i + 1)]
                edge_type_train_t = edge_type_train[batchsize * i:batchsize * (i + 1)]
                edge_index_train_t = edge_index_train[batchsize * i:batchsize * (i + 1)]
            else:
                feature_embedding_train_t = edge_index_emb_trian[batchsize * i:]
                edge_type_train_t = edge_type_train[batchsize * i:]
                edge_index_train_t = edge_index_train[batchsize * i:]

            feature_embedding_train_t = feature_embedding_train_t.to(device)
            edge_index_train_t = edge_index_train_t.to(device)
            edge_type_train_t = edge_type_train_t.to(device)
            output = model(feature_embedding_train_t, edge_index_train_t.t() , edge_type_train_t )
            pre_scores_all.extend(output)
            # loss_train = loss(output[data.train_mask], edge_type[data.train_mask])

        loss_train = loss(torch.stack(pre_scores_all) , edge_type[train_index_l] )
        out = torch.stack(pre_scores_all).max(1)[1].to('cpu').detach().numpy()
        label = edge_type[train_index_l].cpu().detach().numpy()
        acc_train = accuracy_score(out , label  )
        # acc_val = accuracy_score(out , label )
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              '-五折-F_fold: {:04d}'.format(idx + 1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()))
            # 'acc_val: {:.4f}'.format(acc_val.item()), )
        return acc_train

    def test(test_index_l):

        model.eval()
        sample_number = len(test_index_l)
        # batchsize
        pre_scores_all = []

        batch_total = int(sample_number/batchsize) + 1

        edge_index_emb_test = edge_index_emb[test_index_l]
        edge_type_test = edge_type[test_index_l]
        edge_index_test = edge_index.t()[test_index_l]
        for i in range(batch_total-1):
            if batchsize*(i + 2) < sample_number:
                feature_embedding_test_k = edge_index_emb_test[batchsize * i:batchsize*(i + 1)]
                edge_type_test_k = edge_type_test[batchsize * i:batchsize * (i + 1)]
                edge_index_test_k = edge_index_test[batchsize * i:batchsize * (i + 1)]
            else:
                feature_embedding_test_k = edge_index_emb_test[batchsize * i:]
                edge_type_test_k = edge_type_test[batchsize * i:]
                edge_index_test_k = edge_index_test[batchsize * i:]


            output_test = model(feature_embedding_test_k,  edge_index_test_k.t(),edge_type_test_k)
            pre_scores_all.extend(output_test)

        loss_test = loss(torch.stack(pre_scores_all), edge_type_test)
        out = torch.stack(pre_scores_all).max(1)[1].to('cpu').detach().numpy()
        label = edge_type_test.cpu().detach().numpy()
        acc_test = accuracy_score(out, label)
        f1 = f1_score(out, label, average='macro')
        precision = precision_score(out, label, average='macro')
        recall = recall_score(out, label, average='macro')

        # 补充评价数据
        event_num = 4
        y_one_hot = label_binarize(label, classes=np.arange(event_num))
        roc_auc_test = roc_auc_score(y_one_hot, torch.stack(pre_scores_all).cpu().detach().numpy(), average='micro')
        precision_score_test = precision_score(label, out, average='micro')
        recall_score_test = recall_score(label, out, average='micro')
        roc_aupr_score = utils2.roc_aupr_score(y_one_hot, torch.stack(pre_scores_all).cpu().detach().numpy(), average='macro')
        # 记录每个事件的评价指标
        start = time.time()
        result_all, result_eve =  utils2.evaluate(out, pre_scores_all, label, 4)
        # print("time used:", (time.time() - start) / 3600)
        utils2.save_result("Dataset", "all", result_all)
        utils2.save_result("Dataset", "each", result_eve)

        return acc_test, loss_test, f1, precision, recall, roc_auc_test,precision_score_test,recall_score_test,roc_aupr_score,result_all,result_eve

    model.apply(init_weights)


    max_val_acc = 0
    max_acc = 0
    log_path_auc = './log/GraphSage-all.csv'
    file_all = open(log_path_auc, 'a+', encoding='utf-8', newline='')

    log_path_aupr = './log/GraphSage-all.csv'
    file_aupr = open(log_path_aupr, 'a+', encoding='utf-8', newline='')
    file_aupr = open(log_path_aupr, 'a+', encoding='utf-8', newline='')



    for epoch in range(args.epochs):
        kf = StratifiedKFold(n_splits=5, random_state=1996, shuffle = True)  # 初始化KFold
        train_acc_mean = []
        test_acc_mean = []
        test_f1_mean = []
        test_precision_mean = []
        test_recall_mean = []
        test_auc_mean = []
        test_aupr_mean = []
        for idx,(train_index_l, test_index_l) in enumerate( kf.split(edge_index_emb.cpu(),edge_type.cpu())):  # 调用split方法切分数据
            acc_val = train_l(idx,epoch,train_index_l)
            acc_test, loss_test, f1, precision, recall, roc_auc_test,precision_score_test,recall_score_test,roc_aupr_score ,result_all,result_eve = test(test_index_l)
            test_acc_mean.append(acc_test)
            test_f1_mean.append(f1)
            test_precision_mean.append(precision)
            test_recall_mean.append(recall)
            train_acc_mean.append(acc_val)
            test_auc_mean.append(roc_auc_test)
            test_aupr_mean.append(roc_aupr_score)


        # 记录AUC到csv  AUC AUPR ACC  F-score
        csv_writer_all = csv.writer(file_all)
        csv_writer_aupr = csv.writer(file_aupr)
        csv_writer_all.writerow([epoch, '%.4f' %mean(test_auc_mean),'%.4f' %mean(test_aupr_mean),'%.4f' %mean(test_acc_mean),'%.4f' %mean(test_f1_mean)])


        print("五折-Test set results:",
              # "五折-epoch= {:}".format(max_epoch),
              "五折-test_accuracy= {:.4f}".format( mean(test_acc_mean)),
              "五折-precision= {:.4f}".format(mean(test_precision_mean)),
              "五折-recall= {:.4f}".format(mean(test_recall_mean)),
              "五折-f1_score= {:.4f}".format(mean(test_f1_mean)),

              "-roc_auc_test= {:.4f}".format(mean(test_auc_mean)),
              "-precision_score_test= {:.4f}".format(mean(test_precision_mean)),
              "-recall_score_test= {:.4f}".format(mean(test_recall_mean)),
              "-roc_aupr_score= {:.4f}".format(mean(test_aupr_mean))
              )

        logger.info(
            f"-Test set results:-epoch= {epoch}-test_accuracy= {acc_test}-precision= {precision}-recall= {recall}-f1_score= {f1}-roc_auc_test= {roc_auc_test}-precision_score_test= {precision_score_test}-recall_score_test= {recall_score_test}-roc_aupr_score= {roc_aupr_score}")
        logger.info(
            f"-epoch= {epoch}-acc= {result_all[0]}-aupr= {result_all[1]}-auc= {result_all[2]}-f1= {result_all[3]}-precision= {result_all[4]}-recall= {result_all[5]}-label0-aupr= {result_eve[0][1]}-label1-aupr= {result_eve[1][1]}-label2-aupr= {result_eve[2][1]}-label3-aupr= {result_eve[3][1]}"

        )

        if mean(test_acc_mean) > max_acc:
            max_val_acc = acc_val
            max_acc = mean(test_acc_mean)
            max_epoch = epoch + 1
            max_f1 = mean(test_f1_mean)
            max_precision = mean(test_precision_mean)
            max_recall = mean(test_recall_mean)
            max_roc_auc_test = mean(test_auc_mean)
            max_precision_score_test = mean(test_precision_mean)
            max_recall_score_test = mean(test_recall_mean)
            max_roc_aupr_score = mean(test_aupr_mean)

            max_result_all = result_all
            max_result_eve = result_eve

    print("Test set results:",
          "max-epoch= {:}".format(max_epoch),
          "max-test_accuracy= {:.4f}".format(max_acc),
          "max-precision= {:.4f}".format(max_precision),
          "max-recall= {:.4f}".format(max_recall),
          "max-f1_score= {:.4f}".format(max_f1),

    "-max_roc_auc_test= {:.4f}".format(max_roc_auc_test),
    "-max_precision_score_test= {:.4f}".format(max_precision_score_test),
    "-max_recall_score_test= {:.4f}".format(max_recall_score_test),
    "-max_roc_aupr_score= {:.4f}".format(max_roc_aupr_score)
          )
    logger.info(f"Test set results: epoch= {max_epoch} ,test_accuracy= {max_acc},precision= {max_precision},recall= {max_recall},f1_score= {max_f1}")

    logger.info(
          f"-max_roc_auc_test= {max_roc_auc_test},-max_precision_score_test= {max_precision_score_test},-max_recall_score_test= {max_recall_score_test},-max_roc_aupr_score= {max_roc_aupr_score}")

    print(
        "epoch= {:}".format(max_epoch),
          "max-acc= {}".format(max_result_all[0]),
          "max-aupr= {}".format(max_result_all[1]),
          "max-auc= {}".format(max_result_all[2]),
          "max-f1= {}".format(max_result_all[3]),
          "max-precision= {}".format(max_result_all[4]),
          "max-recall= {}".format(max_result_all[5]))
    logger.info(
        f"epoch= {max_epoch}max-acc= {max_result_all[0]}max-aupr= {max_result_all[1]}max-auc= {max_result_all[2]}max-f1= {max_result_all[3]}max-precision= {max_result_all[4]}max-recall= {max_result_all[5]}")

    logger.info(
        f"epoch= {max_epoch}max-label0-aupr= {max_result_eve[0][1]}max-label1-aupr= {max_result_eve[1][1]}max-label2-aupr= {max_result_eve[2][1]}max-label3-aupr= {max_result_eve[3][1]}"

          )
    return max_acc, max_precision, max_recall, max_f1


if __name__ == "__main__":

    t = time.time()
    acc_list =[]
    precision_list = []
    recall_list = []
    f1_list = []

    for i, seed in enumerate(args.random_seed):
        print('\ntraning {}th model'.format(i+1))
        acc, precision, recall, f1 = main(seed)
        acc_list.append(acc*100)
        precision_list.append(precision*100)
        recall_list.append(recall*100)
        f1_list.append(f1*100)

    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
    print('total time:', time.time() - t)





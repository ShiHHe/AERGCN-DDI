import os
import argparse, time, torch, logging, os
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
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from sklearn.model_selection import KFold,StratifiedKFold
from statistics import mean, stdev
from sklearn.model_selection import KFold
import numpy as np

from sklearn.preprocessing import label_binarize

from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from datetime import datetime
from utils2 import setup_logger
import csv

exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M') + '-fold1-task3'
log_file_path = './log/'+exec_name+'.log'
logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
# logger.info(args)

warnings.filterwarnings('ignore')

torch.cuda.device_count()

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='stance', help='detection task of stance or bot')
parser.add_argument('--relation_select', type=int, default=[0,1,2,3], nargs='+', help='selection of relations in the graph (0-6)')
parser.add_argument('--random_seed', type=int, default=[0,1,2,3], nargs='+', help='selection of random seeds')
parser.add_argument('--model', type=str, default='AERGCN', help='GCN, GAT, GraphSage, RGCN, AERGCN')
parser.add_argument('--hidden_dimension', type=int, default=256, help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (1 - keep probability)')
parser.add_argument('--epochs', type=int, default=1200, help='training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='weight decay for optimizer')
args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_acc_a = 0

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)


def main(seed):

    x = utils2.prepare_features()
    edge_index, edge_label = utils2.get_edge_label_attri('./Dataset/DDInter/5-fold/fold-4_kown_data_ddi.txt')
    edge_index_newdrugs, edge_label_newdrugs = utils2.get_edge_label_attri('./Dataset/DDInter/5-fold/fold-4_kown_unknown_data_ddi.txt')

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
    # 构建新药属性特征
    edge_index_emb_newdrugs = []
    for emb in torch.tensor(edge_index_newdrugs).t():
        emb1 = emb[0].item()
        emb2 = emb[1].item()
        edge_index_emb_triple = torch.FloatTensor(torch.cat([x_feature[emb1], x_feature[emb2]], -1) )
        # edge_index_emb.add(edge_index_emb_triple)
        edge_index_emb_newdrugs.append(edge_index_emb_triple)



    edge_index_emb = torch.stack(edge_index_emb)
    datasetDDI = Data(x, edge_index=torch.tensor(edge_index), edge_type=torch.tensor(edge_label),edge_weight=torch.tensor(edge_label))

    data = datasetDDI
    data_test = Data(x, edge_index=torch.tensor(edge_index_newdrugs), edge_type=torch.tensor(edge_label_newdrugs),edge_weight=torch.tensor(edge_label_newdrugs))
    assert args.task in ['stance', 'bot'], "args.task should be choose from ['stance', 'bot']"
    out_dim = 4

    sample_number = len(data.edge_index[1])
    shuffled_idx = shuffle(np.array(range(sample_number)), random_state=seed)
    train_idx = shuffled_idx[:int(0.7 * sample_number)]
    val_idx = shuffled_idx[int(0.7 * sample_number):int(0.9 * sample_number)]
    test_idx = shuffled_idx[int(0.9 * sample_number):]
    data.train_mask = sample_mask(train_idx, sample_number)
    data.val_mask = sample_mask(val_idx, sample_number)
    data.test_mask = sample_mask(test_idx, sample_number)

    test_mask = data.test_mask
    train_mask = data.train_mask
    val_mask = data.val_mask

    data = data.to(device)
    embedding_size = data.x.shape[1]
    relation_num = len(args.relation_select)
    index_select_list = (data.edge_type == 100)

    data_test = data_test.to(device)
    index_select_list_test = (data_test.edge_type == 100)

    relation_dict = {
        0:'followers',
        1:'friends',
        2:'mention',
        3:'reply',
        4:'quoted',
        5:'url',
        6:'hashtag'
    }


    print('relation used:', end=' ')
    for features_index in args.relation_select:
            index_select_list = index_select_list + (features_index == data.edge_type)
            index_select_list_test = index_select_list_test + (features_index == data_test.edge_type)
            print('{}'.format(relation_dict[features_index]), end='  ')
    print('\n')
    edge_index = data.edge_index[:, index_select_list]
    edge_type = data.edge_type[index_select_list]
    edge_weight =  data.edge_weight[index_select_list]

    edge_index_test = data_test.edge_index[:, index_select_list_test]


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

    batchsize =2000
    def train_l(epoch):
        model.train()
        sample_number = len(edge_type)
        pre_scores_all = []
        batch_total = int(sample_number/batchsize) + 1

        for i in range(batch_total-1):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            if batchsize*(i + 2) < sample_number:
                feature_embedding_train = edge_index_emb[batchsize * i:batchsize*(i + 1)]
                edge_type_train = edge_type[batchsize * i:batchsize * (i + 1)]
                edge_index_train = edge_index.t()[batchsize * i:batchsize * (i + 1)]
            else:
                feature_embedding_train = edge_index_emb[batchsize * i:]
                edge_type_train = edge_type[batchsize * i:]
                edge_index_train = edge_index.t()[batchsize * i:]

            feature_embedding_train = feature_embedding_train.to(device)
            edge_index_train = edge_index_train.to(device)
            edge_type_train = edge_type_train.to(device)
            # 此为测试无finger
            # x_feature = torch.zeros(1974, 300, dtype=torch.float32)
            output = model(x_feature, feature_embedding_train, edge_index_train.t() , edge_type_train )
            # output = model(feature_embedding_train, edge_index_train)
            pre_scores_all.extend(output)
            # loss_train = loss(output[data.train_mask], edge_type[data.train_mask])

        loss_train = loss(torch.stack(pre_scores_all) , edge_type )
        out = torch.stack(pre_scores_all).max(1)[1].to('cpu').detach().numpy()
        label = edge_type.cpu().detach().numpy()
        acc_train = accuracy_score(out , label  )
        # acc_val = accuracy_score(out , label )
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()))
            # 'acc_val: {:.4f}'.format(acc_val.item()), )
        logger.info('start training...')
        logger.info(f'Epoch: {epoch + 1}loss_train: {loss_train.item()}acc_train: {acc_train.item()}' )
        return acc_train
    # return 0

    def test():

        model.eval()
        sample_number = len(edge_index_emb_newdrugs)
        # batchsize
        pre_scores_all = []

        batch_total = int(sample_number/batchsize) + 1

        for i in range(batch_total-1):
            if batchsize*(i + 2) < sample_number:
                feature_embedding_test_k = edge_index_emb_newdrugs[batchsize * i:batchsize*(i + 1)]
                edge_type_test_k = edge_label_newdrugs[batchsize * i:batchsize * (i + 1)]
                edge_index_test_k = edge_index_test.t()[batchsize * i:batchsize * (i + 1)]
            else:
                feature_embedding_test_k = edge_index_emb_newdrugs[batchsize * i:]
                edge_type_test_k = edge_label_newdrugs[batchsize * i:]
                edge_index_test_k = edge_index_test.t()[batchsize * i:]

            feature_embedding_test_k = torch.stack(feature_embedding_test_k).to(device)
            edge_index_test_k = edge_index_test_k.t().to(device)
            edge_type_test_k = torch.tensor(edge_type_test_k)
            # 测试
            # x_feature = torch.zeros(1974, 300, dtype=torch.float32)
            output_test = model(x_feature,feature_embedding_test_k,  edge_index_test_k,edge_type_test_k)
            pre_scores_all.extend(output_test)

        loss_test = loss(torch.stack(pre_scores_all), torch.tensor(edge_label_newdrugs).to(device))
        out = torch.stack(pre_scores_all).max(1)[1].to('cpu').detach().numpy()
        label = torch.tensor(edge_label_newdrugs).cpu().detach().numpy()
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
    max_acc_test = 0
    log_path = './log/fold1-task3.csv'
    log_path_aupr = './log/fold1-task3-aupr.csv'
    file = open(log_path, 'a+', encoding='utf-8', newline='')
    file_aupr = open(log_path_aupr, 'a+', encoding='utf-8', newline='')
    best_acc_a=0
    for epoch in range(args.epochs):
        acc_val = train_l(epoch)
        acc_test, loss_test, f1, precision, recall, roc_auc_test,precision_score_test,recall_score_test,roc_aupr_score ,result_all,result_eve= test()
        # 记录AUC到csv
        csv_writer = csv.writer(file)
        # csv_writer.writerow([f'Epoch', 'Accuracy'])
        csv_writer.writerow([epoch, '%.4f' %roc_auc_test,'%.4f' %roc_aupr_score,'%.4f' %acc_test,'%.4f' %f1])

        # save best model
        model_state_file = './cache/' + exec_name + '.pth'
        if best_acc_a < acc_test:
            best_acc_a = acc_test
            best_epoch = epoch
            # logger.info(
            #     f'Epoch {epoch:04d} | Loss {loss.item():.7f} | Best MRR {best_mrr:.4f} | Best epoch {best_epoch:04d}')
            if best_epoch >= 20:
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                torch.save(model, "./cache/model.pkl")


            if (epoch%10 ==0):
                print("-Test set results:",
                  "-epoch= {:}".format(epoch),
                  "-test_accuracy= {:.4f}".format( acc_test),
                  "-precision= {:.4f}".format(precision),
                  "-recall= {:.4f}".format(recall),
                  "-f1_score= {:.4f}".format(f1),

                  "-roc_auc_test= {:.4f}".format(roc_auc_test),
                  "-precision_score_test= {:.4f}".format(precision_score_test),
                  "-recall_score_test= {:.4f}".format(recall_score_test),
                  "-roc_aupr_score= {:.4f}".format(roc_aupr_score)
                  )
            logger.info(f"-Test set results:-epoch= {epoch}-test_accuracy= {acc_test}-precision= {precision}-recall= {recall}-f1_score= {f1}-roc_auc_test= {roc_auc_test}-precision_score_test= {precision_score_test}-recall_score_test= {recall_score_test}-roc_aupr_score= {roc_aupr_score}")

            print(
                        "-epoch= {:}".format(epoch),
                  "-acc= {}".format(result_all[0]),
                  "-aupr= {}".format(result_all[1]),
                  "-auc= {}".format(result_all[2]),
                  "-f1= {}".format(result_all[3]),
                  "-precision= {}".format(result_all[4]),
                  "-recall= {}".format(result_all[5]),

                  "-label0-aupr= {}".format(result_eve[0][1]),
                  "-label1-aupr= {}".format(result_eve[1][1]),
                  "-label2-aupr= {}".format(result_eve[2][1]),
                  "-label3-aupr= {}".format(result_eve[3][1])

                  )
            logger.info(
                        f"-epoch= {epoch}-acc= {result_all[0]}-aupr= {result_all[1]}-auc= {result_all[2]}-f1= {result_all[3]}-precision= {result_all[4]}-recall= {result_all[5]}-label0-aupr= {result_eve[0][1]}-label1-aupr= {result_eve[1][1]}-label2-aupr= {result_eve[2][1]}-label3-aupr= {result_eve[3][1]}"

                  )


        if acc_test > max_acc_test:
            max_acc_test = acc_test
            max_acc = acc_test
            max_epoch = epoch + 1
            max_f1 = f1
            max_precision = precision
            max_recall = recall

            max_roc_auc_test = roc_auc_test
            max_precision_score_test = precision_score_test
            max_recall_score_test = recall_score_test
            max_roc_aupr_score = roc_aupr_score


            max_result_all = result_all
            max_result_eve = result_eve
    file.close()
    print("Test set results:",
          "epoch= {:}".format(max_epoch),
          "test_accuracy= {:.4f}".format(max_acc),
          "precision= {:.4f}".format(max_precision),
          "recall= {:.4f}".format(max_recall),
          "f1_score= {:.4f}".format(max_f1),

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

    print(
        "epoch= {:}".format(max_epoch),
          "max-label0-aupr= {}".format(max_result_eve[0][1]),
          "max-label1-aupr= {}".format(max_result_eve[1][1]),
          "max-label2-aupr= {}".format(max_result_eve[2][1]),
          "max-label3-aupr= {}".format(max_result_eve[3][1])

          )

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
        logger.info(f'traning {i + 1}th model' )
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


    logger.info(f'acc:       {np.array(acc_list).mean()} + { np.std(acc_list)}')
    logger.info(f'precision: {np.array(precision_list).mean()} + {np.std(precision_list)}')
    logger.info(f'recall:    {np.array(recall_list).mean()} + {np.std(recall_list)}')
    logger.info(f'f1:        {np.array(f1_list).mean()} + {np.std(f1_list)}')
    logger.info(f'total time:{time.time() - t}' )




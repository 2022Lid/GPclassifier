from gpclassifier import gp_classifier, count_leaf_nodes
# from a_others.read_data import RD
import random
import numpy as np
from functools import partial
from func_tools import *
from sklearn.model_selection import train_test_split
from read_data import read_arff

dir_name = "../datasets/arff"
GSES = ['GSE14728', 'GSE42408', 'GSE46205', 'GSE76613', 'GSE145709']
resultspath = '../amse-new1/'
cishu = 30

bw = list(range(1, 31))


def main():
    for GSE in GSES:
        print(GSE)

        # 读取数据
        all_datas, flat_class = read_arff(dir_name, GSE)
        all_datas = np.array(all_datas)
        feat_num = len(all_datas[0]) - 1
        instanceNum = len(all_datas)
        auc_sum = 0.0
        feat_sum = 0
        selected_feat_all = []
        selected_feat_num_all = []
        auc_all = []
        num_datas = all_datas[:, :-1]
        num_label = all_datas[:, -1]

        # 随机30次取平均值
        for l in range(cishu):

            # 划分数据集
            train_set, test_set, train_label, test_label = train_test_split(num_datas, num_label,
                                                                            train_size=0.7, test_size=0.3,
                                                                            random_state=bw[l], stratify=num_label)
            train_num = len(train_set)
            test_num = len(test_set)

            # 合并训练集的标签和特征
            data_training = []
            for index, b in enumerate(train_label):
                i = np.append(train_set[index], b)
                i = i.tolist()
                data_training.append(i)

            # 合并测试集标签和特征
            data_testing = []
            for index, b in enumerate(test_label):
                i = np.append(test_set[index], b)
                i = i.tolist()
                data_testing.append(i)

            # 将训练集的少数类多数类分开
            Cmin_train = []
            Cmaj_train = []
            for datas in data_training:
                if datas[-1] == 1:
                    Cmin_train.append(datas)
                elif datas[-1] == 0:
                    Cmaj_train.append(datas)
            min1 = len(Cmin_train)
            maj1 = len(Cmaj_train)

            # 将测试集的少数类多数类分开
            Cmin_test = []
            Cmaj_test = []
            for datas in data_testing:
                if datas[-1] == 1:
                    Cmin_test.append(datas)
                elif datas[-1] == 0:
                    Cmaj_test.append(datas)
            x = len(Cmin_test)
            y = len(Cmaj_test)
            M = [x, y]

            # 训练分类模型
            pareto_first_front, aucc, hof, evalfunc, pset = gp_classifier(Cmin_train, Cmaj_train, train_num, feat_num, instanceNum, data_training, min1, maj1)
            pareto_first_front = sorted(pareto_first_front, key=lambda ind: ind.fitness.values[1], reverse=True)

            # for a, ind in enumerate(pareto_first_front):
            #     print("第%d个" % a, ind, ind.fitness.values)

            # 寻找最好个体
            best_ind = pareto_first_front[0]
            print(best_ind, best_ind.fitness.values)

            # 计算auc
            auc = aucc(best_ind, Cmin=Cmin_test, Cmaj=Cmaj_test, N=M)

            # 测试
            print("第%d次分类auc为" % l, auc)
            selected_feat = []
            selected_feat.append(str(best_ind))
            feat_sum += count_leaf_nodes(best_ind)
            selected_feat_all.append(selected_feat)
            auc_all.append(auc)
            auc_sum += auc
        auc_avg = auc_sum / cishu
        selected_feat_num_avg = feat_sum / cishu

        # 写入结果
        with open(resultspath + GSE + '.txt', 'w') as f:

            f.write('样本数：' + str(len(all_datas)) + '\t原始特征数：' + str(feat_num) + '\n')
            f.write("平均auc:" + str(auc_avg) + '\n最大值:' + str(max(auc_all)) + '\t最小值：' + str(min(auc_all)) +
                    '\t标准差:' + str(np.std(auc_all)) + "平均特征数:" + str(selected_feat_num_avg)+'\n')
            f.write('\n')
            for item in range(len(selected_feat_all)):
                f.write('第%d次：\n' % item)
                for i in selected_feat_all[item]:
                    f.write(str(i) + '\n')
            for index, m in enumerate(auc_all):
                f.write('第%d次：\n' % index)
                f.write(str(m) + '\n')
                f.write('\n')


if __name__ == '__main__':
    main()

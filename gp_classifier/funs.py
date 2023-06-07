from func_tools import distinct


# 重复率
def rept_rate(p):
    return (len(p) - len(distinct(p)))/len(p)


# 训练集上得到的front中的acc最好的个体
def bestone_front(front):
    acc_inds = {}
    # print("训练集上得到的front：")
    for j, ind in enumerate(front):
        acc_inds[ind.fitness.values[0]] = ind
        # print("（1）个体%d及其适应度值为：" % (j + 1), ind.fitness.values, ind)
    # print(acc_inds)
    a = acc_inds[max(acc_inds.keys())]  # 训练集上得到的front中的acc最好的个体
    # print("1训练集上最好个体及其适应度：", a.fitness.values, a)
    return a



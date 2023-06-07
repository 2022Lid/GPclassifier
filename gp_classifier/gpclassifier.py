import operator
import itertools
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from func_tools import *
import numpy as np
from scipy.stats import rankdata
import re


# 统计叶子结点个数
def count_leaf_nodes(ind):
    ind_str = str(ind)
    leaf_nodes = ind_str.count("f")
    return leaf_nodes


# 统计选择出来的特征
def count_selected_feat(ind, pset):
    list = []
    ind_str = str(ind)
    for f in reversed(pset.arguments):
        if ind_str.find(f) != -1:
            list.append(f)
            ind_str = ind_str.replace(f, '')
    list.reverse()
    return list


def gp_classifier(Cmin, Cmaj, train_num, feat_num, instanceNum, data_training, minnum, majnum):
    # 创建一个迭代器，它返回指定次数的对象。如果未指定，则无限返回对象。
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, feat_num), float, "f")

    # 保护性除法
    def Div(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    # 添加操作符
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(Div, [float, float], float)

    # 定义问题
    creator.create("MultiObjMin", base.Fitness, weights=(1.0, -1.0))               # 多目标，一个最小化，一个最大化
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.MultiObjMin)    # 创建individual类

    # 过程中所需参数动态绑定
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # 判断类别
    def classes(func, datas):
        if func(datas[:-1]) >= 0:
            return 1.0
        else:
            return 0.0

    # 对sigmoid函数的优化，避免了出现极大的数据溢出
    def sig(x):
        if x >= 0:
            return 2.0 / (1 + np.exp(-x)) - 1
        else:
            return (2 * np.exp(x)) / (1 + np.exp(x)) - 1

    # amse-attrNum
    def evalfunc(ind, toolbox, Cmin, Cmaj):
        attrNum = count_leaf_nodes(ind)
        func = toolbox.compile(expr=ind)
        try:
            func([])
            return 0,
        except:
            Nmin = len(Cmin)
            Nmaj = len(Cmaj)
            k = [(0.5, Nmin, Cmin), (-0.5, Nmaj, Cmaj)]
            result = []
            for c in k:
                b = list(map(lambda a: pow(sig(func(a[:-1])) - c[0], 2) / (c[1] * 2), c[2]))
                result.append(1 - sum(b))
            result = sum(result) / 2
            return result, attrNum

    def aucc(ind, Cmin, Cmaj, N):
        func = toolbox.compile(expr=ind)
        Pc_min = list(map(lambda a: func(a[:-1]), Cmin))  # 少数类的输出（正类）
        Pc_maj = list(map(lambda a: func(a[:-1]), Cmaj))  # 多数类的输出（负类）
        o = np.array(Pc_min + Pc_maj)
        r = rankdata(o)
        sum_r_min = sum(r[:N[0]])
        auc = (sum_r_min - N[0] * (N[0] + 1) / 2) / (N[0] * N[1])
        return auc

    # 将自定义函数填充到工具箱当中，在之后的算法部分可以调用，适应度评估evaluate, 遗传操作：交叉mate，变异mutate，选择select。
    toolbox.register("evaluate", evalfunc, toolbox=toolbox, Cmin=Cmin, Cmaj=Cmaj)
    toolbox.register("selectGen1", tools.selTournament, tournsize=7)
    toolbox.register('select', tools.emo.selTournamentDCD)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 初始化种群
    N_POP = 500  # 种群数量
    N_GEN = 50  # 进化代数50
    CXPB = 0.8  # 交叉概率
    MUTPB = 0.2  # 突变概率
    pop = toolbox.population(n=N_POP)

    hof = tools.ParetoFront()        # 多目标

    # 创建一个有着两个统计目标的统计对象
    stats_result = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_attrNum = tools.Statistics(lambda ind: ind.fitness.values[1])
    mstats = tools.MultiStatistics(result=stats_result, attrNum=stats_attrNum)

    print("多目标统计器的两个目标为：", mstats.keys())

    # 注册统计目标的统计项
    mstats.register("平均值", numpy.mean)
    mstats.register("标准差", numpy.std)
    mstats.register("最小值", numpy.min)
    mstats.register("最大值", numpy.max)

    # 装饰器的作用就是用一个新函数封装旧函数，然后会返回一个新函数，新函数就叫做装饰器，分别对之前注册的mate，mutate函数进行包装，限制树的最大深度为8
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

    # 开始进化 多目标进化算法
    hof, pop, hofs, hof_fits, rept_rates = algorithms.eaNSGA2_improve(pop, toolbox, CXPB, MUTPB, N_GEN, N_POP, stats=mstats, halloffame=hof, verbose=True)

    fronts = tools.emo.sortNondominated(pop, len(pop))
    # # 使用枚举循环得到各层的标号与pareto解
    # for i, front in enumerate(fronts):
    #     print("pareto非支配等级%d解的个数：" % (i + 1), len(front), front)

    pareto_first_front = fronts[0]                       # 返回的不同的pareto层集合fronts中第一个front为当前最优解集
    pareto_first_front = distinct(pareto_first_front)    # 去除重复的个体

    return pareto_first_front, aucc, hof, evalfunc, pset

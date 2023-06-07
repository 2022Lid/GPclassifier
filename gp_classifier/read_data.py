
#读取arff文件的函数
def read_arff(dir_name, file_name):
    all_datas = []
    with open(dir_name + "/" + file_name + ".arff", "r", encoding='utf-8') as arff_file:
        for line in arff_file:
            if not (line.startswith("@") or line.startswith("%")):
                if line != "\n":
                    line_list = line.strip("\n").split(",")
                    # print(line_list)
                    for i, item in enumerate(line_list[:-1]):
                        if item != "?":
                            line_list[i] = float(item)
                        else:
                            line_list[i] = 0.0
                    all_datas.append(line_list)
    all_label = [datas[-1] for datas in all_datas]
    list_label = sorted(list(set(all_label)), key=all_label.index)
    # print(list_label)
    classes = len(list_label)

    label_num = [all_label.count(l) for l in list_label]  # [20,10]
    d = {}
    for i in range(len(list_label)):
        d[list_label[i]] = label_num[i]
    # print(d.items())
    s = sorted(d.items(), key=lambda a: (a[1], a[0]),reverse = True)
    print(s)
    z = {}
    for i, item in enumerate(s):
        z[item[0]] = i
    for data in all_datas:
        data[-1] = z[data[-1]]
    return all_datas, classes
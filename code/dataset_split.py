import json
import random


def origin_split():

    a = []

    for i in range(1, 175):
        a.append(str(i).zfill(3))
    random.shuffle(a)
    train = [i for i in a[:122]]
    val = [i for i in a[122:157]]
    test = [i for i in a[157:]]

    train.sort()
    val.sort()
    test.sort()
    dic = {"train":train, "val":val, "test":test}
    print(dic)

    with open("data/data/splits.json", 'w', encoding='utf-8') as tf:
        json.dump(dic, tf)



def nnunet_split():
    real_split = []
    
    dic = {"train":[],"val":[]}
    for t in range(1, 1221):
        dic["train"].append(f"IMPLANT_{str(t).zfill(4)}")
    for v in range(1221, 1571) :
        dic["val"].append(f"IMPLANT_{str(v).zfill(4)}")

    for i in range(5) :
        real_split.append(dic)

    with open("data/data_trans/splits_final.json", 'w', encoding='utf-8') as tf:
        json.dump(real_split, tf)


if __name__ == '__main__':
    # origin_split()
    nnunet_split()
import numpy as np

from util.categoryDef import categoryDef
from util.encode_column import encode_column
from util.normalize_column import normalize_column
from util.similarity_matrix import ufrs_kersim, ufrs_kersim_with_missing


def ADIFIE(data, delta, category, missing_value):
    n, m = data.shape

    FE = np.zeros(m)
    FE_x = np.zeros((n, m))
    FRC_x = np.zeros((n, m))
    weight_x = np.zeros((n, m))

    Epsilon = np.zeros(m)

    for j in range(m):
        tmp = [float(e) for e in data[:, j] if e != missing_value]
        if category[j] == 1:
            Epsilon[j] = np.std(tmp, ddof=1) / delta

    ssr_list = []

    for l in range(1, m + 1):
        col = l
        r = np.zeros((n, n))

        for j in range(n):
            a = data[j, col - 1]
            x = data[:, col - 1]

            for k in range(j + 1):
                if j == k:
                    r[k, j] = 1
                else:
                    r[j, k] = ufrs_kersim_with_missing(a, x[k], Epsilon[col - 1], category[col - 1], missing_value)
                    r[k, j] = r[j, k]
        # print(r)
        ssr_list.append(r)
    '''
        第1部分
    '''
    FE_x = np.zeros((n, m))
    FRC_x = np.zeros((n, m))
    weight_x = np.zeros((n, m))
    FE = np.zeros(m)

    for l in range(1, m + 1):
        RM = ssr_list[l - 1]
        FE_tem = np.mean(1 - np.sum(RM, axis=1) / n)
        FE[l - 1] = FE_tem
        t = 1

        while t <= n:
            RM_temp = np.copy(RM)
            RM_temp = np.delete(RM_temp, t - 1, axis=0)
            RM_temp = np.delete(RM_temp, t - 1, axis=1)

            FE_xtem = np.mean(1 - np.sum(RM_temp, axis=1) / (n - 1))
            FE_x[t - 1, l - 1] = FE_xtem
            FRC_x[t - 1, l - 1] = np.sum(RM[t - 1, :]) - np.sum(np.sum(RM_temp)) / (n - 1)
            weight_x[t - 1, l - 1] = np.sqrt(np.sum(RM[t - 1, :]) / n)

            t += 1
    FRC_x[np.isnan(FRC_x)] = n

    RFE = 1 - FE_x / np.tile(FE, (n, 1))
    RFE[RFE < 0] = 0
    RFE[RFE > 1] = 0
    RFE[np.isnan(RFE)] = 0

    FOD_Xl = np.zeros((n, m))

    for r in range(n):
        FOD_temp = RFE[r, :]
        for s in range(m):
            if FRC_x[r, s] > 0:
                FOD_Xl[r, s] = FOD_temp[s] * (n - abs(FRC_x[r, s]))
            else:
                FOD_Xl[r, s] = FOD_temp[s] * (n + abs(FRC_x[r, s]))
    '''
        第2部分
    '''

    e = np.argsort(FE)[::-1]
    L = m

    ssr_deA_list = []

    while L > 0:
        l = m - L
        lA_de = e[:L]
        ssr_de_tmp = ssr_list[lA_de[0]]
        for j in range(L):
            ssr_de_tmpAtemp = ssr_list[lA_de[j]]
            ssr_de_tmp = np.minimum(ssr_de_tmp, ssr_de_tmpAtemp)

        ssr_deA_list.append(ssr_de_tmp)
        L -= 1
    ssr_deA_list = ssr_deA_list[::-1]

    FE_deA = np.zeros(m)
    FE_deA_x = np.zeros((n, m))
    FRC_deA_x = np.zeros((n, m))
    weightA_de = np.zeros((n, m))

    for l in range(1, m + 1):
        RM_deA = ssr_deA_list[l - 1]
        FE_deA_tem = np.mean(1 - np.sum(RM_deA, axis=1) / n)
        FE_deA[l - 1] = FE_deA_tem
        t = 1

        while t <= n:
            RM_deA_temp = np.copy(RM_deA)
            RM_deA_temp = np.delete(RM_deA_temp, t - 1, axis=0)
            RM_deA_temp = np.delete(RM_deA_temp, t - 1, axis=1)

            FE_deA_xtem = np.mean(1 - np.sum(RM_deA_temp, axis=1) / (n - 1))
            FE_deA_x[t - 1, l - 1] = FE_deA_xtem
            FRC_deA_x[t - 1, l - 1] = np.sum(RM_deA[t - 1, :]) - np.sum(np.sum(RM_deA_temp)) / (n - 1)
            weightA_de[t - 1, l - 1] = np.sqrt(np.sum(RM_deA[t - 1, :]) / n)
            t += 1

    FRC_deA_x[np.isnan(FRC_deA_x)] = n
    RFE_deA = 1 - FE_deA_x / np.tile(FE_deA, (n, 1))
    RFE_deA[RFE_deA < 0] = 0
    RFE_deA[RFE_deA > 1] = 0
    RFE_deA[np.isnan(RFE_deA)] = 0
    FOD_deA_Xl = np.zeros((n, m))

    for r in range(n):
        FODA_temp = RFE_deA[r, :]
        for s in range(m):
            if FRC_deA_x[r, s] > 0:
                FOD_deA_Xl[r, s] = FODA_temp[s] * (n - abs(FRC_deA_x[r, s]))
            else:
                FOD_deA_Xl[r, s] = FODA_temp[s] * (n + abs(FRC_deA_x[r, s]))

    FEOF = np.zeros(n)

    for j in range(n):
        temp1 = FOD_Xl[j, :]
        temp2 = FOD_deA_Xl[j, :]
        FEOF[j] = 1 - ((np.sum((1 - temp1) * weight_x[j, :]) + np.sum((1 - temp2) * weightA_de[j, :])) / (2 * m))

    return FEOF


if __name__ == '__main__':

    trandata = np.array([[0.0000, 0.625, 0],
                         [2.0000, "*", 1],
                         [3.0000, 0.0000, 0.75],
                         ["*", 1.0000, 0.125],
                         [1.0000, 0.5, "*"],
                         [0.0000, 0.25, 0.5]])
    trandata = np.array([["A", 6, 0.1],
                         ["C", "*", 0.9],
                         ["D", 1, 0.7],
                         ["*", 9, 0.3],
                         ["B", 5, "*"],
                         ["A", 3, 0.5]])
    category = np.array(categoryDef(trandata, '*'))
    l = np.where(category == 1)[0].tolist()
    n, m = trandata.shape
    for i in range(m):
        if i in l:
            trandata[:, i] = normalize_column(trandata[:, i], '*')
        else:
            trandata[:, i] = encode_column(trandata[:, i], '*')
    print(category)
    print(category)

    print(ADIFIE(trandata, 1, [3, 1, 1], '*'))

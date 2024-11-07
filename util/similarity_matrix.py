# 计算相似矩阵
def ufrs_kersim(a, x, e):
    if abs(float(a) - float(x)) > e:
        kersim = 0
    else:
        if e == 0:
            if a == x:
                kersim = 1
            else:
                kersim = 0
        else:
            kersim = 1 - abs(a - x)

    return kersim


# 计算相似矩阵
def ufrs_kersim_with_missing(a, x, e, category, missing_value):
    # category 代表每列的类型 1 为数值，2为布尔，3为类别
    if category == 1:
        kersim = kersim_numeric(a, x, e, missing_value)
    else:
        kersim = kersim_nominal(a, x, missing_value)

    return kersim


def kersim_numeric(a, x, e, missing_value):
    if a == missing_value or x == missing_value:
        kersim = 1
        return kersim
    if abs(float(a) - float(x)) > e:
        kersim = 0
    else:
        kersim = 1 - abs(float(a) - float(x))
    return kersim


def kersim_nominal(a, x, missing_value):
    if a == missing_value or x == missing_value:
        kersim = 0
        return kersim
    if a == x:
        kersim = 1
    else:
        kersim = 0
    return kersim

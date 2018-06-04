import opt


def unnormalize(x):
    return x * opt.STD + opt.MEAN

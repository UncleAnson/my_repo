from numpy import *


def difcost(a, b):
    dif = 0
    for i in range(shape(a)[0]):
        for j in range(shape(a)[1]):
            # 计算欧几里得距离
            dif += pow(a[i, j] - b[i, j], 2)
    return dif


def factorize(v, pc=10, iter=1000):
    ic = shape(v)[0]
    fc = shape(v)[1]

    # Initialize the weight and feature matrices with random values
    w = matrix([[random.random() for j in range(pc)] for i in range(ic)])
    h = matrix([[random.random() for i in range(fc)] for i in range(pc)])

    # Perform operation a maximum of iter times
    firstin = True
    for i in range(iter):
        wh = w * h

        # Calculate the current difference
        # and if
        tmp = difcost(v, wh)
        if not firstin and (cost - tmp) / cost < 0.001:
            break
        else:
            cost = tmp
            firstin = False
            print(cost)

        # Terminate if the matrix has been fully factorized
        if cost == 0:
            break

        # Update feature matrix
        hn = (transpose(w) * v)
        hd = (transpose(w) * w * h)

        h = matrix(array(h) * array(hn) / array(hd))

        # Update weights matrix
        wn = (v * transpose(h))
        wd = (w * h * transpose(h))

        w = matrix(array(w) * array(wn) / array(wd))

    return w, h

if __name__ == '__main__':
    l1 = [[1,2,3],[4,5,6]]
    m1 = matrix(l1)
    l2 = [[1,2],[3,4],[5,6]]
    m2 = matrix(l2)
    print(m1,'\n',m2)
    print(shape(m1))# (2,3)
    print(m1*m2)
    w,h = factorize(m1*m2, pc=3,iter=100)
    print(w)
    # [[0.30174692 0.63429855 0.5591728]
    #  [0.01901229 0.95937879 1.80866631]]
    print(h)
    # [[20.71320809  7.26808042]
    #  [2.13984768 17.95460822]
    # [25.73900441 25.78504757]]
    print(w*h)  # == m1*m2
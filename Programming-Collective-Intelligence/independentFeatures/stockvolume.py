import nmf
import datetime
from numpy import *
import pickle

tickers = ['YHOO', 'AVP', 'BIIB', 'BP', 'CL', 'CVX',
           'PG', 'XOM', 'AMGN']

l1 = pickle.load(open('input/finance.txt', 'rb'))
dates = pickle.load(open('input/date.txt', 'rb'))
print('Data loaded!')
print('Start training!')

w, h = nmf.factorize(matrix(l1), pc=5)
print(matrix(l1))
print(w*h)
print(w)
print(h)

for i in range(shape(h)[0]):  # 特征矩阵
    print("Feature %d" % i)

    # Get the top stocks for this feature
    ol = [(h[i, j], tickers[j]) for j in range(shape(h)[1])]
    ol.sort()
    ol.reverse()
    for j in range(len(tickers)):
        print(ol[j])
    print()

    # Show the top dates for this feature
    porder = [(w[d, i], d) for d in range(300)]
    porder.sort()
    porder.reverse()
    print([(p[0], dates[p[1]]) for p in porder[0:3]])
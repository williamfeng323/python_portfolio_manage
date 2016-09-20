# import sys
# import operator
# import portfolioopt as port
# import pandas

# def sortKey(list):
#     return list[0]
#
# def sortTrail(list):
#     if 'alice' in list:
#         print 'Hello Alice'
#     print sorted(list, key=sortKey)
#
# def main():
#     name = 'world'
#     if len(sys.argv) > 1:
#         name_list = sys.argv[1:]
#         name = ''
#         for val in name_list:
#             name += '%s ' % val
#     print 'hello %s' % name
#     sortTrail(name_list)
#     dic = {'a':3,'b':2,'c':1}
#     # print dic
#     print sorted(dic.items(),key=operator.itemgetter(1))
import os
import pandas as pd
import sys
import cvxopt as opt
import numpy as np

def loadData(file_name):
    file_path = os.path.join('./data',file_name)
    returns = pd.read_csv(file_path, index_col=0)
    cov_mat = returns.cov()
    avg_returns = returns.mean()
    return returns, cov_mat, avg_returns

def main():
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = 'stockReturns.csv'
        returns, cov_mat, avg_returns = loadData(file_name)

if __name__ == '__main__':
    main()

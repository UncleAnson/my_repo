import networkx as nx
G = nx.random_graphs.barabasi_albert_graph(1000,3)  #生成一个n=1000，m=3的BA无标度网络

print(G.degree(0))  #返回某个节点的度
print(G.degree())  #返回所有节点的度
print(nx.degree_histogram(G))  #返回图中所有节点的度分布序列（从1至最大度的出现频次）
# print(nx.diameter(G))  #网络直径

import matplotlib.pyplot as plt  #导入科学绘图的matplotlib包
degree =  nx.degree_histogram(G)  #返回图中所有节点的度分布序列
# x = range(len(degree))  #生成x轴序列，从1到最大度
# y = [z / float(sum(degree)) for z in degree]
# plt.loglog(x, y, color="blue",linewidth=2)  #在双对数坐标轴上绘制度分布曲线
# plt.show()

def ave(degree):#平均度计算
    s_um = 0
    for i in range(len(degree)):
        s_um =s_um+i*degree[i]
    return s_um/nx.number_of_nodes(G)
print(ave(degree))

def p(degree):
    x = list(range(len(degree)))
    y = [i for i in degree]
    plt.bar(x, y, align='center')#plot、loglog
    plt.ylim(0, 300)
    plt.title('Distribution of Nodes')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    for a, b in zip(x,y):
        plt.text(a, b+2,'%.0f' % b, ha='center')
    #plt.savefig("degree.png")
    plt.show()
p(degree)

# 群聚系数
print(nx.average_clustering(G))  # 平均群聚系数的计算
print(nx.clustering(G)) # 计算各个节点的群聚系数
# 直径和平均距离
print(nx.diameter(G)) # 返回图G的直径（最长的最短路径的长度）
print(nx.average_shortest_path_length(G))  #返回图G所有节点间平均最短路径长度
# 匹配性
print(nx.degree_assortativity(G))  # 图的度匹配性

# 中心性
# .........


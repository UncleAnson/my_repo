import pandas as pd
import numpy as np

# 构建 用户-电影 矩阵
def convert(data, num_users, num_movies):
    new_data = []
    for id_user in range(1, num_users+1): # 用户编号
        id_movie = data[:, 1][data[:, 0]==id_user] # 某位用户的参评电影
        id_rating = data[:, 2][data[:, 0]==id_user] # 某位用户的所有评分
        # 该用户未参评的电影评分以0表示
        ratings = np.zeros(num_movies, dtype=np.uint32)
        ratings[id_movie-1]=id_rating   # id_movie从1开始计数
        # 如果出现的user_id在当前数据集中未评论过（都是0），则跳过
        if sum(ratings)==0:
            continue
        new_data.append(ratings)
    return new_data # shape: [num_user, num_movie]


def get_dataset_1M(ROOT_DIR):
    # 分别读取训练和测试数据
    # Falling back to the 'python' engine because the 'c' engine does not support regex separators
    training_set = pd.read_csv(ROOT_DIR+'/ml-1m/train.dat', sep='::', header=None, engine='python', encoding='latin-1')
    test_set = pd.read_csv(ROOT_DIR+'/ml-1m/test.dat', sep='::', header=None, engine='python', encoding='latin-1')
    training_set = np.array(training_set, dtype=np.uint32)
    test_set = np.array(test_set, dtype=np.uint32)

    num_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))  # 有6040个用户
    num_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) # 有3952部电影

    # 分别构建不同数据集的用户-电影矩阵
    training_set = convert(training_set, num_users, num_movies)
    test_set = convert(test_set, num_users, num_movies)
    return training_set, test_set


def _get_dataset(ROOT_DIR):
    return get_dataset_1M(ROOT_DIR)

if __name__ == '__main__':
    train, test = _get_dataset('./ROOT_DIR')
    print(train)
    print('==============================')
    print(test)

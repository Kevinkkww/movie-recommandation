import datetime
import pickle
import math
import pandas as pd
import os
import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    movies_df = pd.read_csv('ml-25m/movies.csv')
    ratings_df = pd.read_csv('ml-25m/ratings.csv')
    tags_df = pd.read_csv('ml-25m/tags.csv')


    if os.path.exists('data/') != True:
        os.mkdir('data/')

    if os.path.exists('data/last_liked_tags/') != True:
        os.mkdir('data/last_liked_tags/')

    # 取出用户最近一个喜欢的电影（打分>=4.0）移出原数据，作为标签使用
    ratings_df_copy = ratings_df.copy()
    tags_df_copy = tags_df.copy()

    users_list = list(set(ratings_df_copy.userId))
    users_list.sort()
    print(len(users_list))

    ratings_index_list = []
    tags_index_list = []

    last_ratings_df = pd.DataFrame() #用于将所有用户最后一个喜欢的电影信息存入last_liked_ratings.csv文件中

    counter = 0

    for user in users_list:
        try:
            temp_df = ratings_df_copy[ratings_df_copy.userId == user].copy()
            temp_df = temp_df[temp_df.rating >= 4] #取用户喜欢的所有电影

            last_time = max(temp_df.timestamp)

            temp_df = temp_df[temp_df.timestamp == last_time]

            if len(temp_df) > 1: #若存在相同时间戳，可取最后一个数据
                temp_df = temp_df.iloc[[len(temp_df) - 1]]

            ratings_index_list.append(temp_df.index.values[0])

            if counter == 0:
                last_ratings_df = temp_df
                counter = 1

            else:
                last_ratings_df = pd.concat([last_ratings_df, temp_df], ignore_index=True)

        except Exception:
            ratings_index_list.append(ratings_df_copy[ratings_df_copy.userId == user].index.values[
                                          0])

        try:#取用户给最后一部喜欢电影的评论信息（大多数为空）
            temp_df = tags_df_copy[tags_df_copy.userId == user].copy()
            temp_df = temp_df[temp_df.rating >= 4]
            last_movie = temp_df.movieId.values[0]
            temp_df = temp_df[temp_df.movieId == last_movie]

            if len(temp_df) == 0:
                continue

            else:
                temp_df.to_csv('data/last_liked_tags/' + str(user) + '.csv',
                               index=False)
                tags_index_list.extend(list(
                    temp_df.index.values))

        except Exception:
            pass

    last_ratings_df.to_csv('data/last_liked_ratings.csv', index=False)

    #将用户喜欢的电影移出并保存新的评分和评论文件
    ratings_df_removed = ratings_df_copy.drop(ratings_index_list)
    tags_df_removed = tags_df_copy.drop(tags_index_list)

    ratings_df_removed.to_csv('data/ratings_df_last_liked_movie_removed.csv', index=False)
    tags_df_removed.to_csv('data/tags_df_last_liked_movie_removed.csv', index=False)

    #采用简单的NLP方法处理评论，包括全部小写、去括号......（需要使用正则表达式等工具）
    #最终得到tags_df_mod.csv文件将用户、电影以及处理后的评论联系起来
    tags_df_removed = pd.read_csv('data/tags_df_last_liked_movie_removed.csv')

    tags_df_mod = tags_df_removed.copy().drop('timestamp', axis=1).dropna()
    tags_df_mod['tag'] = tags_df_mod['tag'].str.lower()

    for index, row in tags_df_mod.iterrows():
        tag = row.tag

        correct_tag = re.sub(r' \([^)]*\)', '',
                             tag)

        if 'based' in correct_tag:
            tags_df_mod.loc[index, 'tag'] = correct_tag
            continue

        if '-' in correct_tag:
            tags_df_mod.loc[index, 'tag'] = correct_tag
            continue

        if re.findall(r'\b\w{2}\b', correct_tag):
            tags_df_mod.loc[
                index, 'tag'] = np.NaN

        elif re.findall(r'\b\w{1}\b', correct_tag):
            tags_df_mod.loc[index, 'tag'] = np.NaN

        elif tag == correct_tag:
            continue

        else:
            tags_df_mod.loc[index, 'tag'] = correct_tag
            pass

    tags_df_mod = tags_df_mod.dropna()

    tags_df_mod.to_csv('data/tags_df_mod.csv', index=False)

    #得到每个电影对应的常见评论存入data/movie_tags文件夹中，记录评论及其出现次数
    if os.path.exists('data/movie_tags/') != True:
        os.mkdir('data/movie_tags/')

    tags_df_mod = pd.read_csv('data/tags_df_mod.csv')
    # 具体用户信息对于本模块无效所以删除userId列
    tags_df_no_user = tags_df_mod.copy().drop('userId', axis=1)

    movieId_list = list(set(tags_df_no_user.movieId))
    for movieId in movieId_list:
        df_select = tags_df_no_user[tags_df_no_user.movieId == movieId].copy().drop('movieId', axis=1)
        df_select['COUNT'] = 1
        df_select_group = df_select.groupby(['tag']).count()
        df_select_group = df_select_group.sort_values(by=['COUNT'], ascending=False).reset_index()
        df_select_group.to_csv('data/movie_tags/' + str(movieId) + '.csv', index=False)

    #得到每个用户对应的常见评论存入data/user_tags文件夹中，记录评论及其出现次数
    if os.path.exists('data/user_tags/') != True:
        os.mkdir('data/user_tags/')

    tags_df_mod = pd.read_csv('data/tags_df_mod.csv')
    #具体电影信息对本模块无效所以删除movieId列
    tags_df_user = tags_df_mod.copy().drop('movieId', axis=1)

    userId_list = list(set(tags_df_user.userId))
    for userId in userId_list:
        df_select = tags_df_user[tags_df_user.userId == userId].copy().drop('userId', axis=1)
        df_select['COUNT'] = 1
        df_select_group = df_select.groupby(['tag']).count()
        df_select_group = df_select_group.sort_values(by=['COUNT'], ascending=False).reset_index()
        df_select_group.to_csv('data/user_tags/' + str(userId) + '.csv', index=False)

    #创建常见（被35个以上用户使用过）评论信息
    tags_df_mod = pd.read_csv('data/tags_df_mod.csv')
    common_tags_df = tags_df_mod.groupby(['tag']).count().sort_values('userId', ascending=False).copy().drop('movieId',
                                                                                                             axis=1).reset_index()
    common_tags_df = common_tags_df[common_tags_df.userId >= 35]
    common_tags_df.to_csv('data/common_tags.csv', index=False)

    '''
    处理movie.csv文件存入movies_mod.csv中作为输入数据
    输入特征包括movieId，title，year，观看次数，平均得分+/-标准差，平均得分，电影类型（多列）
    '''
    ratings_df_removed = pd.read_csv('data/ratings_df_last_liked_movie_removed.csv')
    movies_df_mod = movies_df.copy()

    movies_df_mod['YEAR'] = 0
    movies_df_mod['UPPER_STD'] = 0
    movies_df_mod['LOWER_STD'] = 0
    movies_df_mod['AVG_RATING'] = 0
    movies_df_mod['VIEW_COUNT'] = 0

    genres_list = []
    for index, row in movies_df.iterrows():
        try:
            genres = row.genres.split('|')
            genres_list.extend(genres)
        except:
            genres_list.append(row.genres)

    genres_list = list(set(genres_list))
    genres_list.remove('IMAX')
    genres_list.remove('(no genres listed)')  # Replace with 'None'
    genres_list.append('None')

    for genre in genres_list:
        movies_df_mod[genre] = 0

    for index, row in movies_df_mod.iterrows():
        movieId = row.movieId
        title = row.title

        try:
            genres = row.genres.split(
                '|')
        except Exception:
            genres = list(row.genres)


        try:
            matcher = re.compile(
                '\(\d{4}\)')
            parenthesis_year = matcher.search(title).group(0)
            matcher = re.compile('\d{4}')
            year = matcher.search(parenthesis_year).group(0)

            movies_df_mod.loc[index, 'YEAR'] = int(year)

        except Exception:
            pass

        try:
            ratings_df_select = ratings_df_removed[
                ratings_df_removed.movieId == movieId]
            std = np.std(ratings_df_select.rating)
            average_rating = np.mean(ratings_df_select.rating)

            upper_std = average_rating + std

            if upper_std > 5:
                upper_std = 5

            lower_std = average_rating - std

            if lower_std < 0.5:
                lower_std = 0.5

            view_count = len(ratings_df_select)

            movies_df_mod.loc[index, 'UPPER_STD'] = upper_std
            movies_df_mod.loc[index, 'LOWER_STD'] = lower_std
            movies_df_mod.loc[index, 'AVG_RATING'] = average_rating
            movies_df_mod.loc[index, 'VIEW_COUNT'] = view_count

        except Exception:
            pass

        if 'IMAX' in genres:
            genres.remove('IMAX')

        if '(no genres listed)' in genres:
            genres.remove('(no genres listed)')
            genres.append('None')

        for genre in genres:
            movies_df_mod.loc[index, genre] = 1

    movies_df_mod = movies_df_mod[movies_df_mod.YEAR != 0]
    movies_df_mod = movies_df_mod[movies_df_mod.VIEW_COUNT != 0]

    movies_df_mod.to_csv('data/movies_mod.csv', index=False)

    #将ratings数据和movie数据结合起来
    movies_df_mod = pd.read_csv('data/movies_mod.csv')
    ratings_df_removed = pd.read_csv('data/ratings_df_last_liked_movie_removed.csv')
    ratings_movies_df = ratings_df_removed.merge(movies_df_mod, how='left',
                                                 on='movieId').dropna()

    #计数用户喜欢电影类型并转化成百分比表示
    #百分比可以消去用户数据量大影响，对于一个给很多电影大国评论的用户来说
    #他的计数会远高于用户数据量小的用户，但并不能表示他更喜欢该标签
    #为了防止模型权重被歪曲，统一使用百分比
    users_list = list(set(ratings_movies_df.userId))
    total_user_like_df = pd.DataFrame()
    total_user_dislike_df = pd.DataFrame()

    progress_counter_1 = 0
    progress_counter_2 = .10

    for user in users_list:
        temp_df = ratings_movies_df[ratings_movies_df.userId == user]
        like_df = temp_df[temp_df.rating >= 4].iloc[:, 14:]  #只选取类型信息
        dislike_df = temp_df[temp_df.rating < 4].iloc[:, 14:]

        liked_total_counts = 0
        liked_dict = {'userId': user, 'War': 0, 'Animation': 0, 'Horror': 0, 'Sci-Fi': 0, 'Fantasy': 0, 'Thriller': 0,
                      'Crime': 0, 'Mystery': 0,
                      'Documentary': 0, 'Children': 0, 'Action': 0, 'Adventure': 0, 'Musical': 0, 'Film-Noir': 0,
                      'Drama': 0,
                      'Romance': 0, 'Comedy': 0, 'Western': 0, 'None': 0}

        disliked_total_counts = 0
        disliked_dict = {'userId': user, 'War': 0, 'Animation': 0, 'Horror': 0, 'Sci-Fi': 0, 'Fantasy': 0,
                         'Thriller': 0, 'Crime': 0, 'Mystery': 0,
                         'Documentary': 0, 'Children': 0, 'Action': 0, 'Adventure': 0, 'Musical': 0, 'Film-Noir': 0,
                         'Drama': 0,
                         'Romance': 0, 'Comedy': 0, 'Western': 0, 'None': 0}

        progress_counter_1 += 1
        if progress_counter_1 / len(users_list) >= progress_counter_2:
            print(progress_counter_1 / len(users_list) * 100, '%')
            progress_counter_2 += .10

        for genre in list(like_df.columns):
            if len(like_df) == 0:
                pass

            else:
                liked_total_counts += sum(like_df[genre])

            if len(dislike_df) == 0:
                pass

            else:
                disliked_total_counts += sum(dislike_df[genre])

        for genre in list(like_df.columns):
            if liked_total_counts == 0:
                pass

            else:
                liked_genre_total_counts = sum(like_df[genre])
                liked_dict[genre] = liked_genre_total_counts / liked_total_counts

            if disliked_total_counts == 0:
                pass

            else:
                disliked_genre_total_counts = sum(dislike_df[genre])
                disliked_dict[genre] = disliked_genre_total_counts / disliked_total_counts

        user_like_df = pd.DataFrame(liked_dict, index=[
            0])
        user_dislike_df = pd.DataFrame(disliked_dict, index=[0])

        #串联所有用户的计数，将结果存入data/total_user_like_df.csv和data/total_user_dislike_df.csv中
        if len(total_user_like_df) == 0:
            total_user_like_df = user_like_df
        else:
            total_user_like_df = pd.concat([total_user_like_df, user_like_df], ignore_index=True)

        if len(total_user_dislike_df) == 0:
            total_user_dislike_df = user_dislike_df
        else:
            total_user_dislike_df = pd.concat([total_user_dislike_df, user_dislike_df], ignore_index=True)

    total_user_like_df.to_csv('data/total_user_like_df.csv', index=False)
    total_user_dislike_df.to_csv('data/total_user_dislike_df.csv', index=False)

    #本模块运行极为耗时（RTX2080Ti耗时2天左右）
    #创建用户喜欢/不喜欢的标签并序列化，保存结果
    if os.path.exists('data/final/') != True:
        os.mkdir('data/final/')

    common_tags = pd.read_csv('data/common_tags.csv', index_col= False)
    tags = list(set(common_tags.tag))

    vector_counter = 0
    vectorized_dict = {}

    for tag in tags:
        vectorized_dict[tag] = vector_counter
        vector_counter += 1

    ratings_df_removed = pd.read_csv('data/ratings_df_last_liked_movie_removed.csv')

    user_list = list(set(ratings_df_removed.userId))
    like_dislike_tags = pd.DataFrame()
    index_counter = 0
    progress_counter_1 = 0
    progress_counter_2 = 5

    #耗时过长，打印开始时间信息方便用户确认
    start_time = datetime.datetime.now()
    print('Start Time:', start_time)

    for user in user_list:
        progress_counter_1 += 1

        temp_ratings_df = ratings_df_removed[ratings_df_removed.userId == user]
        like_tags_df = pd.DataFrame()
        dislike_tags_df = pd.DataFrame()

        for index, row in temp_ratings_df.iterrows():  #为每个用户创建标签
            try:
                if row.rating >= 4:
                    temp_movie_df = pd.read_csv('data/movie_tags/{}.csv'.format(str(int(row.movieId))))

                    if len(like_tags_df) == 0:
                        like_tags_df = temp_movie_df
                    else:
                        like_tags_df = pd.concat([like_tags_df, temp_movie_df], ignore_index= True)

                else:
                    temp_movie_df = pd.read_csv('data/movie_tags/{}.csv'.format(str(int(row.movieId))))

                    if len(like_tags_df) == 0:
                        dislike_tags_df = temp_movie_df
                    else:
                        dislike_tags_df = pd.concat([dislike_tags_df, temp_movie_df], ignore_index= True)
            except Exception:
                pass

        try: #将给所有电影都打低分/高分的用户剔除防止误导模型
            like_tags_list = list(like_tags_df.tag)
            dislike_tags_list = list(dislike_tags_df.tag)
        except Exception:
            continue

        like_dict = {}
        dislike_dict = {}

        for tag in like_tags_list:
            like_dict[tag] = like_tags_list.count(tag) * -1

        for tag in dislike_tags_list:
            dislike_dict[tag] = dislike_tags_list.count(tag) * -1

        #对标签计数并进行排序
        like_tags_counted = sorted(like_dict, key= lambda tag: like_dict[tag])
        dislike_tags_counted = sorted(dislike_dict, key= lambda tag: dislike_dict[tag])

        like_tags_vectorized = []
        dislike_tags_vectorized = []

        if len(like_tags_counted) < 50:
            num_like_tags = len(like_tags_counted)
        else:
            num_like_tags = 50

        if len(dislike_tags_counted) < 50:
            num_dislike_tags = len(like_tags_counted)
        else:
            num_dislike_tags = 50

        for tag in like_tags_counted[:num_like_tags]:
            try:
                tag_vector = vectorized_dict[tag]
                like_tags_vectorized.append(tag_vector)
            except Exception:
                pass

        for tag in dislike_tags_counted[:num_dislike_tags]:
            try:
                tag_vector = vectorized_dict[tag]
                dislike_tags_vectorized.append(tag_vector)
            except Exception:
                pass

        if len(like_tags_vectorized) < 20 or len(dislike_tags_vectorized) < 20:
            continue

        like_dislike_dict = {}

        like_dislike_dict['userId'] = user

        for x in range(20):
            like_dislike_dict['LIKE_' + str(x)] = like_tags_vectorized[x]
            like_dislike_dict['DISLIKE_' + str(x)] = dislike_tags_vectorized[x]

        concat_df = pd.DataFrame(like_dislike_dict, index=[0])

        if len(like_dislike_tags) == 0:
            like_dislike_tags = concat_df

        else:
            like_dislike_tags = pd.concat([like_dislike_tags, concat_df], ignore_index= True)

        if (progress_counter_1 / len(user_list)) * 100 >= progress_counter_2:
            print((progress_counter_1 / len(user_list)) * 100, '% completed')
            print('Processing Time:', datetime.datetime.now() - start_time)
            print('Current Time:', datetime.datetime.now())
            progress_counter_2 += 5

    like_dislike_tags = like_dislike_tags.astype('int64')
    like_dislike_tags.to_csv('data/final/like_dislike_tags.csv', index = False)
    with open('data/vectorized_dict.pkl', 'wb') as writer:
        pickle.dump(vectorized_dict, writer)

    #创建电影和标签的联系并保存结果
    if os.path.exists('data/final/') != True:
        os.mkdir('data/final/')

    movies_df_mod = pd.read_csv('data/movies_mod.csv')
    movieId_list = list(movies_df_mod.movieId)
    del movies_df_mod

    movie_tags_df = pd.DataFrame()
    index_counter = 0

    with open('data/vectorized_dict.pkl', 'rb') as reader:
        vectorized_dict = pickle.load(reader)

    for movie in movieId_list:

        try:
            temp_df = pd.read_csv('data/movie_tags/{}.csv'.format(
                movie))

            if len(temp_df) < 5:
                continue

            vectorized_tag = []
            movie_tags = list(temp_df.tag)

            for tag in movie_tags:
                try:
                    tag_vector = vectorized_dict[tag]
                    vectorized_tag.append(tag_vector)
                except Exception:
                    pass

            if len(vectorized_tag) < 5:
                continue

            movie_tags_df.loc[index_counter, 'movieId'] = movie

            for x in range(5):
                movie_tags_df.loc[index_counter, 'TAG_' + str(x)] = vectorized_tag[x]

            index_counter += 1

        except Exception:
            pass
    movie_tags_df.to_csv('data/final/movie_tags_df.csv', index=False)


    #本函数给出模型的统计学信息
    def stats(predictions, true, flex_range=0.5):
        predictions_list = []
        round_list = np.arange(0.5, 5.5, 0.5)

        #将回归模型预测的连续小数转化为离他最近的阶梯值(步长0.5)
        for value in predictions:
            value_ori = value
            compare_diff = 99999
            value_round = 0

            for rating in round_list:
                compare_value = abs(value_ori - rating)

                if compare_value < compare_diff:
                    compare_diff = compare_value
                    value_round = rating

            predictions_list.append(value_round)

        prediction_dict = {'PREDICTION': predictions_list, 'TRUE': list(true)}
        prediction_compare_df = pd.DataFrame(prediction_dict)

        prediction_length = len(prediction_compare_df)

        #针对实际情况进行调整，由于用户打分以0.5为阶梯，预测误差在0.5范围内均应视为预测正确
        #因此引入_flex满足实际情况
        rating_accuracy_flex = 0
        like_dislike_tp_flex = 0
        like_dislike_tn_flex = 0
        like_dislike_fp_flex = 0
        like_dislike_fn_flex = 0

        progress_counter = 0

        for index, row in prediction_compare_df.iterrows():
            #引入误差量之后的计数过程，与上面类似
            predict_like_flex = 0
            true_like_flex = 0

            if row.PREDICTION >= 3.5:
                predict_like_flex = 1

            if row.TRUE >= 3.5:
                true_like_flex = 1

            #在误差范围内即可认为准确预测
            if row.PREDICTION >= (row.TRUE - flex_range) and row.PREDICTION <= (row.TRUE + flex_range):
                rating_accuracy_flex += 1

            if predict_like_flex == true_like_flex:
                if predict_like_flex == 1:
                    like_dislike_tp_flex += 1

                else:
                    like_dislike_tn_flex += 1

            else:
                if predict_like_flex == 1:
                    like_dislike_fp_flex += 1

                else:
                    like_dislike_fn_flex += 1

            progress_counter += 1
            if progress_counter % 1000000 == 0:
                print(str(progress_counter / prediction_length * 100) + '%')

        rating_accuracy_flex = rating_accuracy_flex / prediction_length
        like_dislike_accuracy_flex = (like_dislike_tp_flex + like_dislike_tn_flex) / prediction_length

        print('True Positive: {}, True Negative: {}, False Positive {}, False Negative {}'.format(
            like_dislike_tp_flex, like_dislike_tn_flex, like_dislike_fp_flex, like_dislike_fn_flex))
        print('Rating Accuracy: {}, Catagorical Accuracy (Like/Dislike) {}'.format(rating_accuracy_flex,
                                                                                             like_dislike_accuracy_flex))
        return


    def merge_shuffle_split(split=1):
        print("Generate begin")
        movies_df_mod = pd.read_csv('data/movies_mod.csv')
        ratings_df_removed = pd.read_csv('ml-25m/all.csv')

        selection_range = int(len(ratings_df_removed) * (split))
        ratings_df_removed = ratings_df_removed.iloc[: selection_range, :]

        ratings_df_removed = ratings_df_removed.merge(movies_df_mod, how='left', on='movieId').dropna()
        del movies_df_mod

        total_user_like_df = pd.read_csv('data/total_user_like_df.csv')

        like_columns = list(total_user_like_df.columns)
        like_columns_modified = []

        for column in like_columns:
            if column == 'userId':
                like_columns_modified.append('userId')
            else:
                modify_column = 'user_like_' + column
                like_columns_modified.append(modify_column)

        total_user_like_df.columns = like_columns_modified

        ratings_df_removed = ratings_df_removed.merge(total_user_like_df, how='left', on='userId').dropna()
        del total_user_like_df

        total_user_dislike_df = pd.read_csv('data/total_user_dislike_df.csv')

        dislike_columns = list(total_user_dislike_df.columns)
        dislike_columns_modified = []

        for column in dislike_columns:
            if column == 'userId':
                dislike_columns_modified.append('userId')
            else:
                modify_column = 'user_dislike_' + column
                dislike_columns_modified.append(modify_column)

        total_user_dislike_df.columns = dislike_columns_modified

        ratings_df_removed = ratings_df_removed.merge(total_user_dislike_df, how='left', on='userId').dropna()

        del total_user_dislike_df

        movie_tags_df = pd.read_csv('data/final/movie_tags_df.csv')
        ratings_df_removed = ratings_df_removed.merge(movie_tags_df, how='left', on='movieId').dropna()
        del movie_tags_df

        like_dislike_tags = (pd.read_csv('data/final/like_dislike_tags.csv')).astype('int64')
        ratings_df_removed = ratings_df_removed.merge(like_dislike_tags, how='left', on='userId').dropna()
        del like_dislike_tags

        like_columns_modified.remove('userId')
        dislike_columns_modified.remove('userId')
        like_columns.remove('userId')

        genres_like = ratings_df_removed.loc[:, like_columns_modified]
        genres_dislike = ratings_df_removed.loc[:, dislike_columns_modified]
        genres_movie = ratings_df_removed.loc[:, like_columns]

        rf_columns = []
        for x in range(20):
            rf_columns.append('LIKE_' + str(x))
            rf_columns.append('DISLIKE_' + str(x))
        for x in range(5):
            rf_columns.append('TAG_' + str(x))

        rf_input = ratings_df_removed.loc[:, rf_columns]

        ratings = list(ratings_df_removed.rating)

        ratings_with_ID = ratings_df_removed[['userId','movieId','rating']]
        ratings_with_ID.to_csv('data/ratings_with_Id.csv', index=False)

        del ratings_df_removed


        return genres_like, genres_dislike, genres_movie, rf_input, ratings

    #Genres模型，使用神经网络
    user_liked_genres = keras.Input(shape=(19,))
    user_disliked_genres = keras.Input(shape=(19,))
    movie_genres = keras.Input(shape=(19,))

    liked_input = keras.layers.Dense(19, activation='relu')(user_liked_genres)
    liked_hidden_1 = keras.layers.Dense(50, activation='relu')(liked_input)
    liked_hidden_2 = keras.layers.Dense(50, activation='relu')(liked_hidden_1)

    disliked_input = keras.layers.Dense(19, activation='relu')(user_disliked_genres)
    disliked_hidden_1 = keras.layers.Dense(50, activation='relu')(disliked_input)
    disliked_hidden_2 = keras.layers.Dense(50, activation='relu')(disliked_hidden_1)

    movie_input = keras.layers.Dense(19, activation='relu')(movie_genres)
    movie_hidden_1 = keras.layers.Dense(50, activation='relu')(movie_input)
    movie_hidden_2 = keras.layers.Dense(50, activation='relu')(movie_hidden_1)

    merged_model = keras.layers.concatenate([liked_hidden_2, disliked_hidden_2, movie_hidden_2])
    merged_model_hidden_1 = keras.layers.Dense(150, activation='relu')(merged_model)
    merged_model_hidden_2 = keras.layers.Dense(75, activation='relu')(merged_model_hidden_1)
    merged_model_hidden_3 = keras.layers.Dense(50, activation='relu')(merged_model_hidden_2)

    output_rating = keras.layers.Dense(1, activation='sigmoid')(merged_model_hidden_3)

    genres_model = keras.Model(inputs=[user_liked_genres, user_disliked_genres, movie_genres], outputs=output_rating)

    genres_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['accuracy'],)


    if os.path.exists('models/') != True:
        os.mkdir('models/')

    genres_like, genres_dislike, genres_movie, rf_input, ratings = merge_shuffle_split()

    ratings_with_Id = pd.read_csv('data/ratings_with_Id.csv')
    ratings = np.array(pd.read_csv('data/ratings.csv').values.flatten()).tolist()
    genres_like = pd.read_csv('data/genres_like.csv')
    genres_dislike = pd.read_csv('data/genres_dislike.csv')
    genres_movie = pd.read_csv('data/genres_movie.csv')
    rf_input = pd.read_csv('data/rf_input.csv')

    train_split = 0.8
    split_index = int(len(ratings) * train_split)

    genres_like_train = genres_like.iloc[: split_index, :]
    genres_like_test = genres_like.iloc[split_index:, :]
    del genres_like

    genres_dislike_train = genres_dislike.iloc[: split_index, :]
    genres_dislike_test = genres_dislike.iloc[split_index:, :]
    del genres_dislike

    genres_movie_train = genres_movie.iloc[: split_index, :]
    genres_movie_test = genres_movie.iloc[split_index:, :]
    del genres_movie

    ratings_with_Id_test = ratings_with_Id.iloc[split_index:, :]
    ratings_with_Id_test.to_csv('data/ratings_with_Id_test.csv', index=False)

    ratings_scaled = np.array(ratings) / 5
    ratings_scaled_train = ratings_scaled[: split_index]
    ratings_scaled_test = ratings_scaled[split_index:]

    batch_size = 300
    epochs = 10

    def scheduler(epoch):
        if epoch < 5:
            return 0.001
        else:
            return 0.001 * math.exp(0.1 * (5 - epoch))


    Learning_Rate_Callback = keras.callbacks.LearningRateScheduler(scheduler)


    class Save_Progress_Callback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):  ## Saving and printing after each epoch
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            print("Epoch {}, loss is {:7.3f}, validation loss is {:7.3f}, learning rate is {}.".format(epoch,
                                                                                                       logs["loss"],
                                                                                                       logs["val_loss"],
                                                                                                       lr))


    genres_model.fit(x=[genres_like_train, genres_dislike_train, genres_movie_train],
                     y=ratings_scaled_train,
                     epochs=epochs, verbose=0, batch_size=batch_size, validation_split=0.1, shuffle=True,
                     callbacks=[Learning_Rate_Callback, Save_Progress_Callback()])

    genres_model.save('models/genres_model.h5', overwrite=True, include_optimizer=True)

    genres_model = tf.keras.models.load_model('models/genres_model.h5', compile=True)

    genres_model_predictions = (genres_model.predict(
        x=[genres_like_test, genres_dislike_test, genres_movie_test])) * 5

    print('genres Model Stats:')
    stats(genres_model_predictions, ratings_scaled_test * 5)

    print("print genres result")
    print(type(genres_model_predictions))
    pd.Series(genres_model_predictions.tolist()).to_csv('data/genres_test_predict', index=False)


    rf_input_train = rf_input.iloc[: split_index, :]
    rf_input_test = rf_input.iloc[split_index:, :]

    ratings_train = ratings[: split_index]
    ratings_test = ratings[split_index:]

    random_forest = RandomForestRegressor(n_estimators=80, max_features='sqrt', verbose=2, random_state=True,
                                          n_jobs=-1)
    random_forest.fit(rf_input_train, ratings_train)
    pickle.dump(random_forest, open('tags_model.sav', 'wb'))

    random_forest = pickle.load(open('tags_model.sav', 'rb'))
    print("RF score:")
    print(random_forest.score(rf_input_test, ratings_test))

    random_forest_predict = random_forest.predict(rf_input_test)
    print('Tags Model Stats:')
    stats(random_forest_predict, ratings_test)

    pd.Series(random_forest_predict.tolist()).to_csv('data/tags_test_predict', index=False)

    # 创建组合模型输入
    genres_model_predictions_list = []
    for prediction in genres_model_predictions:
        genres_model_predictions_list.append(prediction[0])

    merged_predictions = pd.DataFrame({'genres_model': genres_model_predictions_list,
                                       'tag_model': list(random_forest_predict),
                                       'genres_true': list(np.array(list(ratings_scaled_test)) * 5),
                                       'tag_true': ratings_test},
                                      index=list(range(len(ratings_test))))

    # Using a linear regression for predictions adjustment:
    X = merged_predictions.loc[:, ['genres_model', 'tag_model']]
    y = np.array(merged_predictions.loc[:, 'genres_true'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10,shuffle=False)

    line_reg = LinearRegression(n_jobs=-1).fit(X_train, y_train)
    pickle.dump(line_reg, open('combine_model.sav', 'wb'))

    # line_reg = pickle.load(open('combine_model.sav', 'rb'))
    print('Linear Regression R2:', line_reg.score(X_test, y_test))
    line_reg_predictions = line_reg.predict(X_test)

    #将所有超出边界的值设为最大/小值
    line_reg_predictions_rounded = []
    for prediction in line_reg_predictions:
        rounded = prediction
        if rounded > 5:
            rounded = 5
        elif rounded < 0.5:
            rounded = 0.5
        line_reg_predictions_rounded.append(rounded)

    #获取Label
    line_reg_predictions_label = []
    for prediction in line_reg_predictions:
        if prediction >= 3.5:
            rounded = 1
        else:
            rounded = 0
        line_reg_predictions_label.append(rounded)

    print("Combine Model Stats:")
    stats(line_reg_predictions_rounded, y_test)
    print(type(line_reg_predictions_rounded))

    pd.Series(line_reg_predictions_rounded).to_csv('data/combine_test_predict', index=False)

    fpr_1, tpr_1, threshold_1 = roc_curve(line_reg_predictions_label, y_test/5)
    roc_auc_1 = auc(fpr_1, tpr_1)
    print(roc_auc_1)
    plt.figure(figsize=(8, 5))

    plt.plot(fpr_1, tpr_1, color='darkorange',  ###假正率为横坐标，真正率为纵坐标做曲线
             lw=1, label='LR (area = %0.3f)' % roc_auc_1, linestyle='-')  # linestyle为线条的风格（共五种）,color为线条颜色

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.05])  # 横竖增加一点长度 以便更好观察图像
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("hyh.png", dpi=600)  # 保存图片，dpi设置分辨率
    plt.show()

    error=0
    def top_10_recommendations(userId):
        # Loading all the datasets needed:
        movies_df_mod = pd.read_csv('data/movies_mod.csv')
        ratings_df_removed = pd.read_csv('data/ratings_df_last_liked_movie_removed.csv')

        # Gathering all the movies in the dataset:
        not_watched = list(movies_df_mod.movieId)

        # Selecting all movies that have not been seen by the user:
        ratings_df_removed = ratings_df_removed[ratings_df_removed.userId == userId]

        if len(ratings_df_removed) == 0:
            return print('User {} does not have enough information. 1'.format(userId)),None

        ratings_df_removed = ratings_df_removed.merge(movies_df_mod, how='left', on='movieId').dropna()

        if len(ratings_df_removed) == 0:
            return print('User {} does not have enough information. 2'.format(userId))

        watched = list(ratings_df_removed.movieId)
        del ratings_df_removed

        for movie in watched:
            if movie in not_watched:
                not_watched.remove(movie)

        # Loading in users' like and disliked genres:
        total_user_like_df = pd.read_csv('data/total_user_like_df.csv')
        total_user_dislike_df = pd.read_csv('data/total_user_dislike_df.csv')

        total_user_like_df = total_user_like_df[total_user_like_df.userId == userId]

        if len(total_user_like_df) == 0:
            return print('User {} does not have enough information. 3'.format(userId))

        total_user_dislike_df = total_user_dislike_df[total_user_dislike_df.userId == userId]
        if len(total_user_dislike_df) == 0:
            return print('User {} does not have enough information. 4'.format(userId))


        like_columns = list(total_user_like_df.columns)
        like_columns_modified = []

        for column in like_columns:
            if column == 'userId':
                like_columns_modified.append('userId')
            else:
                modify_column = 'user_like_' + column
                like_columns_modified.append(modify_column)

        total_user_like_df.columns = like_columns_modified

        dislike_columns = list(total_user_dislike_df.columns)
        dislike_columns_modified = []

        for column in dislike_columns:
            if column == 'userId':
                dislike_columns_modified.append('userId')
            else:
                modify_column = 'user_dislike_' + column
                dislike_columns_modified.append(modify_column)

        total_user_dislike_df.columns = dislike_columns_modified

        movie_tags_df = pd.read_csv('data/final/movie_tags_df.csv')
        like_dislike_tags = (pd.read_csv('data/final/like_dislike_tags.csv')).astype('int64')

        template_df = pd.DataFrame({'movieId': not_watched},
                                   index=list(range(len(not_watched))))
        template_df = template_df.merge(movies_df_mod, how='left', on='movieId').dropna()
        template_df = template_df.merge(movie_tags_df, how='left', on='movieId').dropna()
        del movie_tags_df

        like_dislike_tags = like_dislike_tags[like_dislike_tags.userId == userId]

        if len(like_dislike_tags) == 0:
            global error
            error=1
            return print('User {} does not have enough information. 5'.format(userId)),None

        template_df['userId'] = userId
        template_df = template_df.merge(total_user_like_df, how='left', on='userId').dropna()
        del total_user_like_df
        template_df = template_df.merge(total_user_dislike_df, how='left', on='userId').dropna()
        del total_user_dislike_df
        template_df = template_df.merge(like_dislike_tags, how='left', on='userId').dropna()
        del like_dislike_tags

        like_columns_modified.remove('userId')
        dislike_columns_modified.remove('userId')
        like_columns.remove('userId')

        rf_columns = []
        for x in range(20):
            rf_columns.append('LIKE_' + str(x))
            rf_columns.append('DISLIKE_' + str(x))
        for x in range(5):
            rf_columns.append('TAG_' + str(x))

        genres_like_input = template_df.loc[:, like_columns_modified]
        genres_dislike_input = template_df.loc[:, dislike_columns_modified]
        genres_movie_input = template_df.loc[:, like_columns]

        tags_input = template_df.loc[:, rf_columns]

        movieId_list = list(template_df.movieId)

        del template_df


        genres_model = tf.keras.models.load_model('models/genres_model.h5', compile=True)
        tags_model = pickle.load(open('tags_model.sav', 'rb'))
        combine_model = pickle.load(open('combine_model.sav', 'rb'))


        genres_model_predictions = (genres_model.predict(x=[genres_like_input, genres_dislike_input,
                                                            genres_movie_input])) * 5
        tags_model_predictions = tags_model.predict(tags_input)

        genres_model_predictions_list = []

        for prediction in genres_model_predictions:
            genres_model_predictions_list.append(prediction[0])

        combine_input = pd.DataFrame({'genres_predictions': genres_model_predictions_list,
                                      'tags_predictions': tags_model_predictions},
                                     index=list(range(len(genres_model_predictions))))

        combine_model_predictions = combine_model.predict(combine_input)


        combine_model_predictions_rounded = []

        for prediction in combine_model_predictions:
            rounded = prediction
            if rounded > 5:
                rounded = 5
            elif rounded < 0.5:
                rounded = 0.5

            combine_model_predictions_rounded.append(rounded)


        predictions_df = pd.DataFrame({'movieId': movieId_list,
                                       'genres_predictions': genres_model_predictions_list,
                                       'tags_predictions': tags_model_predictions,
                                       'combine_predictions': combine_model_predictions_rounded},
                                      index=list(range(len(movieId_list))))

        best_movies_df = predictions_df.sort_values(by=['combine_predictions'], ascending=False).iloc[:10, :]

        best_movies_df = best_movies_df.merge(movies_df_mod, how='left', on='movieId').dropna()
        del movies_df_mod

        return predictions_df, best_movies_df

    if os.path.exists('predictions/') != True:
        os.mkdir('predictions/')

    if os.path.exists('predictions/full_predictions') != True:
        os.mkdir('predictions/full_predictions')

    if os.path.exists('predictions/top_10') != True:
        os.mkdir('predictions/top_10')

    ratings_df_removed = pd.read_csv('data/ratings_df_last_liked_movie_removed.csv')
    userId_list = list(set(ratings_df_removed.userId))
    del ratings_df_removed

    userId_list.sort()

    for user in userId_list:
        predictions_df, best_movies_df = top_10_recommendations(user)
        if(error==1):
            error=0
            continue

        print("UserID:", user)
        predictions_df.to_csv('predictions/full_predictions/full_predictions - {}.csv'.format(user), index = False)
        best_movies_df.to_csv('predictions/top_10/top_10 - {}.csv'.format(user), index = False)
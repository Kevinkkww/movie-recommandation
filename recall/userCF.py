import random

import math
import csv
from operator import itemgetter

class UserCF():
    # 初始化相关参数
    def __init__(self):

        self.trainSet = {}
        self.testSet = {}
        self.temp = {}
        self.result = {}
        self.movies = []
        self.i = 0

        self.user_similar_matrix = {}

        self.read_file()

    def read_file(self):
        with open("sort_ratings.csv", 'r') as f:
            for i, line in enumerate(f):
                if i > 2000000:
                    break

                line = line.strip('\n')
                user, movie, rating, timestamp = line.split(',')
                if random.random() < 0.8:
                    self.trainSet.setdefault(user, {})
                    self.trainSet[user][movie] = {'rating': rating, 'time': int(timestamp)}


        with open("id.csv", 'r') as f:
            for i, line in enumerate(f):
                line = line.strip('\n')
                user, movie, rating = line.split(',')
                self.testSet.setdefault(user,[])
                self.testSet[user].append(movie)

        with open("movies_mod.csv",'r') as f:
            for i, line in enumerate(f):
                line = line.strip('\n')
                movieId,Thriller,Documentary,War,Musical,Crime,Drama,Horror,Adventure,Children,Sci_Fi,Comedy,Mystery,Western,Film_Noir,Fantasy,Animation,Action,Romance,Non= line.split(',')
                self.movies.append(movieId)

        print('文件读取完毕')

    # 计算用户之间的相似度
    def calc_user_sim(self):
        K = 10

        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)

        self.movie_count = len(movie_user)
        print(self.movie_count)

        print('计算相似矩阵1')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_similar_matrix.setdefault(u, {})
                    self.user_similar_matrix[u].setdefault(v, 0)
                    self.user_similar_matrix[u][v] += 1 / (1 + abs(self.trainSet[u][movie]['time'] - self.trainSet[v][movie]['time']))

        print('计算相似矩阵2')
        for u, related_users in self.user_similar_matrix.items():
            for v, cuv in related_users.items():
                self.user_similar_matrix[u][v] = cuv / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))


        for user in self.user_similar_matrix:
            temp = sorted(self.user_similar_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]
            self.user_similar_matrix[user] = {}
            for item in temp:
                self.user_similar_matrix[user][item[0]] = item[1]

    # 对单一用户进行推荐
    def recommend_one(self, user):
        print(self.i)
        self.i = self.i + 1

        N = 10
        user_movie_recommand = {}

        if user not in self.trainSet:
            return {}

        if user not in self.user_similar_matrix:
            return {}

        watched_movies = self.trainSet[user]

        for similar_user, similar_factor in self.user_similar_matrix[user].items():
            for movie in self.trainSet[similar_user]:
                if movie in watched_movies:
                    continue
                user_movie_recommand.setdefault(movie, 0)
                user_movie_recommand[movie] += similar_factor * float(self.trainSet[similar_user][movie]['rating'])

        user_movie_recommand = sorted(user_movie_recommand.items(), key=itemgetter(1), reverse=True)[0:N]

        return user_movie_recommand

    # 对所有用户进行推荐
    def recommand_all(self):
        print('开始评估')

        for i, user, in enumerate(self.testSet):
            if user not in self.temp:
                self.temp.setdefault(user,{})
                rec_movies = self.recommend_one(user)
                self.temp[user] = rec_movies

        for user, recommand in self.temp.items():
            if user in self.result:
                continue

            self.result.setdefault(user, [])
            if not recommand:
                while (len(self.result[user]) < 10):
                    movie = random.choice(self.movies)
                    if movie not in self.result[user]:
                        self.result[user].append(movie)

            for item in recommand:
                if item[0] not in self.movies:
                    continue
                self.result[user].append(item[0])
            while (len(self.result[user]) < 10):
                movie = random.choice(self.movies)
                if movie not in self.result[user]:
                    self.result[user].append(movie)


        with open('userCFResult.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)

            for user in self.result:
                print(user)
                csv_writer.writerow([user, self.result[user][0],self.result[user][1],self.result[user][2],self.result[user][3],self.result[user][4],self.result[user][5],self.result[user][6],self.result[user][7],self.result[user][8],self.result[user][9]])


if __name__ == '__main__':
    userCF = UserCF()
    userCF.calc_user_sim()
    userCF.recommand_all()

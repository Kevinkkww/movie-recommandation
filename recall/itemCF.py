import random

import math
import csv
from operator import itemgetter


class MovieCF():
    def __init__(self):

        self.trainSet = {}
        self.testSet = {}

        self.temp = {}
        self.i = 0
        self.movies = []
        self.result = {}

        self.movie_similar_matrix = {}
        self.movie_popular_matrix = {}

        self.read_file()

    def read_file(self):
        with open("sort_ratings.csv", 'r') as f:
            for i, line in enumerate(f):
                if i > 2000000:
                    break
                line = line.strip('\n')
                user, movie, rating, timestamp = line.split(',')
                if random.random() < 0.8:
                    self.trainSet.setdefault(movie, {})
                    self.trainSet[movie][user] = {'rating': rating, 'time': int(timestamp)}

        with open("id.csv", 'r') as f:
            for i, line in enumerate(f):
                line = line.strip('\n')
                user, movie, rating = line.split(',')
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = {'rating': rating}

        with open("movies_mod.csv", 'r') as f:
            for i, line in enumerate(f):
                line = line.strip('\n')
                movieId, Thriller, Documentary, War, Musical, Crime, Drama, Horror, Adventure, Children, Sci_Fi, Comedy, Mystery, Western, Film_Noir, Fantasy, Animation, Action, Romance, Non = line.split(
                    ',')
                self.movies.append(movieId)

        print('文件读取完毕')

    def calc_movie_sim(self):
        K = 20

        print("juzhen1")
        for movie, users in self.trainSet.items():
            if movie not in self.movie_popular_matrix:
                self.movie_popular_matrix[movie] = 0
            length = len(users)
            for u1 in users:
                for u2 in users:
                    if u1 == u2:
                        continue
                    self.movie_popular_matrix[movie] += 1 / ((math.log(1 + length * 1.0)) * (1 + abs(self.trainSet[movie][u1]['time'] - self.trainSet[movie][u2]['time'])))

        self.movie_count = len(self.movie_popular_matrix)

        print("juzhen2")
        for m1,users in self.trainSet.items():
            for m2,users in self.trainSet.items():
                if m1 == m2:
                    continue

                self.movie_similar_matrix.setdefault(m1, {})
                self.movie_similar_matrix[m1].setdefault(m2, 0)

                if self.movie_popular_matrix[m1] == 0 or self.movie_popular_matrix[m2] == 0:
                    self.movie_similar_matrix[m1][m2] = 0
                else:
                    self.movie_similar_matrix[m1][m2] = 1 / math.sqrt(self.movie_popular_matrix[m1] * self.movie_popular_matrix[m2])


        for movie in self.movie_similar_matrix:
            temp = sorted(self.movie_similar_matrix[movie].items(), key=itemgetter(1), reverse=True)[0:K]
            self.movie_similar_matrix[movie] = {}
            for item in temp:
                self.movie_similar_matrix[movie][item[0]] = item[1]

    # 对单一用户进行推荐
    def recommend_one(self, user):
        print(self.i)
        self.i = self.i + 1

        N = 10
        x = 0
        user_movie_recommand = {}

        watched_movies = self.testSet[user]

        for movie, attribute in watched_movies.items():
            if movie not in self.movie_similar_matrix:
                continue

            for related_movie, similar_factor in self.movie_similar_matrix[movie].items():
                if x > N:
                    break
                x = x + 1

                if related_movie in watched_movies:
                    continue
                user_movie_recommand.setdefault(related_movie, 0)
                user_movie_recommand[related_movie] += similar_factor * float(attribute['rating'])

        temp = sorted(user_movie_recommand.items(), key=itemgetter(1), reverse=True)
        return temp


    # 对所有用户进行推荐
    def recommand_all(self):

        for i, user, in enumerate(self.testSet):
            if user not in self.temp:
                self.temp.setdefault(user, {})
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

        with open('itemCFResult.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)

            for user in self.result:
                print(user)
                csv_writer.writerow(
                    [user, self.result[user][0], self.result[user][1], self.result[user][2], self.result[user][3],
                     self.result[user][4], self.result[user][5], self.result[user][6], self.result[user][7],
                     self.result[user][8], self.result[user][9]])


if __name__ == '__main__':
    itemCF = MovieCF()
    itemCF.calc_movie_sim()
    itemCF.recommand_all()
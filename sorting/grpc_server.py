from concurrent import futures
import time
import grpc
import pandas as pd
from io import StringIO

import HelloWorld_pb2
import HelloWorld_pb2_grpc
import movie_pb2
import movie_pb2_grpc
import tensorflow as tf

# 实现 proto 文件中定义的 GreeterServicer
class Greeter(HelloWorld_pb2_grpc.GreeterServicer):
    # 实现 proto 文件中定义的 rpc 调用
    def SayHello(self, request, context):
        print("shoudao")
        print(request.name)
        # 连接 rpc 服务器
        channel = grpc.insecure_channel('172.20.10.7:50051')  # lichenboIP
        # 调用 rpc 服务
        stub = movie_pb2_grpc.GreeterStub(channel)
        a = movie_pb2.MovieRequest(userId=request.name)
        response = stub.getMovie(a)

        print("From Li Chenbo received:\n ")
        print(response.movieList)
        print(response.userDislike)
        print(response.userLike)
        print(response.movieMod)
        movie_temp = pd.read_csv(StringIO(response.movieList),header=None)
        user_dislike = pd.read_csv(StringIO(response.userDislike),header=None)
        user_like = pd.read_csv(StringIO(response.userLike),header=None)
        movie_list = pd.read_csv(StringIO(response.movieMod),header=None)
        movie_temp = pd.DataFrame(movie_temp.values.T, index=movie_temp.columns, columns=movie_temp.index)
        print("generate finish")
        print(movie_temp)
        print(user_like)
        print(user_dislike)
        print(movie_list)

        prediction = (order_model.predict(x=[user_like, user_dislike, movie_list]))
        movie_temp['score'] = prediction
        movie_temp.columns=['movieID','score']
        movie_temp = movie_temp.sort_values(by='score',ascending=False)
        print(movie_temp)

        print(movie_temp.iloc[0,0])
        recMovieList = 'Recommend Movie:{} {} {} {} {}'.format(movie_temp.iloc[0,0],movie_temp.iloc[1,0],movie_temp.iloc[2,0],
                                                               movie_temp.iloc[3,0],movie_temp.iloc[4,0])
        return HelloWorld_pb2.HelloReply(message = recMovieList)

    def SayHelloAgain(self, request, context):
        return HelloWorld_pb2.HelloReply(message='hello {msg}'.format(msg = request.name))

def serve():
    # 启动 rpc 服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    HelloWorld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50060')
    server.start()
    try:
        while True:
            time.sleep(60*60*24) # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    order_model = tf.keras.models.load_model('order_model.h5', compile=True)
    print("load finish")
    serve()
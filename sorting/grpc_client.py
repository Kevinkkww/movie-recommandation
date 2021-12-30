import grpc
import HelloWorld_pb2
import HelloWorld_pb2_grpc


def run():
    # 连接 rpc 服务器
    channel = grpc.insecure_channel('localhost:50060')  # lichenboIP
    # 调用 rpc 服务
    stub = HelloWorld_pb2_grpc.GreeterStub(channel)
    a=HelloWorld_pb2.HelloRequest(name='35147')
    response = stub.SayHello(a)
    print(response.message)

if __name__ == '__main__':
    run()
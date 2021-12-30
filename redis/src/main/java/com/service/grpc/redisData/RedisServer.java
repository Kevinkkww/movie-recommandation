package com.service.grpc.redisData;


import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.redisData.GreeterGrpc;
import io.grpc.redisData.MovieReply;
import io.grpc.redisData.MovieRequest;
import io.grpc.stub.StreamObserver;

import java.io.IOException;

public class RedisServer {
    private int port = 50051;
    private Server server;

    private void start() throws IOException {
        server = ServerBuilder.forPort(port)
                .addService(new RedisServer.GreeterImpl())
                .build()
                .start();

        System.out.println("service start...");

        Runtime.getRuntime().addShutdownHook(new Thread() {

            @Override
            public void run() {

                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                RedisServer.this.stop();
                System.err.println("*** server shut down");
            }
        });
    }

    private void stop() {
        if (server != null) {
            server.shutdown();
        }
    }

    // block 一直到退出程序
    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    // 实现 定义一个实现服务接口的类
    private class GreeterImpl extends GreeterGrpc.GreeterImplBase {

        public void getMovie(MovieRequest req, StreamObserver<MovieReply> responseObserver) {
            System.out.println("service:" + req.getUserId());
            String userid = req.getUserId();
            MovieReply message = ProtoUtils.getMessage(Integer.valueOf(userid));
            responseObserver.onNext(message);
            responseObserver.onCompleted();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        final RedisServer server = new RedisServer();
        server.start();
        server.blockUntilShutdown();
    }
}

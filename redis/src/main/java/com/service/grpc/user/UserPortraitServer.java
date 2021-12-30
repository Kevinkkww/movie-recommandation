package com.service.grpc.user;

import com.service.RedisUtils;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import io.grpc.userPortrait.UserPortraitReply;
import io.grpc.userPortrait.*;


import java.io.IOException;

public class UserPortraitServer {
    private int port = 50051;
    private Server server;

    private void start() throws IOException {
        server = ServerBuilder.forPort(port)
                .addService(new GreeterImpl())
                .build()
                .start();

        System.out.println("service start...");

        Runtime.getRuntime().addShutdownHook(new Thread() {

            @Override
            public void run() {

                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                UserPortraitServer.this.stop();
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


    public static void main(String[] args) throws IOException, InterruptedException {
        final UserPortraitServer server = new UserPortraitServer();
        server.start();
        server.blockUntilShutdown();
    }


    // 实现 定义一个实现服务接口的类
    private class GreeterImpl extends UserPortraitGreeterGrpc.UserPortraitGreeterImplBase {

        public void getUserPortrait(UserPortraitRequest req, StreamObserver<UserPortraitReply> responseObserver) {
            System.out.println("service:" + req.getUserId());
            String userid = req.getUserId();

            UserPortraitReply.Builder reply = UserPortraitReply.newBuilder();
            reply.setUserId(req.getUserId());



            responseObserver.onNext(reply.build());
            responseObserver.onCompleted();
        }
    }

}

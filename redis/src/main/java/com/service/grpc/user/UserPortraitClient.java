package com.service.grpc.user;


import com.google.protobuf.Message;
import com.googlecode.protobuf.format.JsonFormat;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.moviePortrait.MoviePortraitReply;
import io.grpc.moviePortrait.MoviePortraitRequest;
import io.grpc.userPortrait.UserPortraitGreeterGrpc;
import io.grpc.userPortrait.UserPortraitReply;
import io.grpc.userPortrait.UserPortraitRequest;

import java.util.concurrent.TimeUnit;

//grpc客户端类
public class UserPortraitClient {
    private final ManagedChannel channel;
    private final UserPortraitGreeterGrpc.UserPortraitGreeterBlockingStub blockingStub;
    private static final String host = "127.0.0.1";
    private static final int ip = 50051;

    public UserPortraitClient(String host, int port) {
        //usePlaintext表示明文传输，否则需要配置ssl
        //channel  表示通信通道
        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
        //存根
        blockingStub = UserPortraitGreeterGrpc.newBlockingStub(channel);
    }

    public UserPortraitClient(ManagedChannel channel, UserPortraitGreeterGrpc.UserPortraitGreeterBlockingStub blockingStub) {
        this.channel = channel;
        this.blockingStub = blockingStub;
    }


    public void shutdown() throws InterruptedException {

        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public void testResult(String user_id) {
        UserPortraitRequest request = UserPortraitRequest.newBuilder().setUserId(user_id).build();
        UserPortraitReply response = blockingStub.getUserPortrait(request);
        Message someProto = response.getDefaultInstanceForType();


    }

    public static void main(String[] args) {
        UserPortraitClient client = new UserPortraitClient(host, ip);
        for (int i = 0; i <= 5; i++) {
            client.testResult(String.valueOf(i));
        }
    }
}



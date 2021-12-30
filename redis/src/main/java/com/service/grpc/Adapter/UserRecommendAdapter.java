package com.service.grpc.Adapter;

import io.grpc.redisData.UserRecommendReply;

import java.util.Map;

public class UserRecommendAdapter implements GrpcAdapter {
    @Override
    public Map.Entry<byte[], Object> getInstance(String[] items) {
        UserRecommendReply.Builder builder = UserRecommendReply.newBuilder();
        builder.setUserId(Integer.valueOf(items[0]));
        builder.setMovieId1(Integer.valueOf(items[1]));
        builder.setMovieId2(Integer.valueOf(items[2]));
        builder.setMovieId3(Integer.valueOf(items[3]));
        builder.setMovieId4(Integer.valueOf(items[4]));
        builder.setMovieId5(Integer.valueOf(items[5]));
        builder.setMovieId6(Integer.valueOf(items[6]));
        builder.setMovieId7(Integer.valueOf(items[7]));
        builder.setMovieId8(Integer.valueOf(items[8]));
        builder.setMovieId9(Integer.valueOf(items[9]));
        builder.setMovieId10(Integer.valueOf(items[10]));
        UserRecommendReply instance = builder.build();
        return Map.entry(("item_result_" + items[0]).getBytes(), instance);
    }
}

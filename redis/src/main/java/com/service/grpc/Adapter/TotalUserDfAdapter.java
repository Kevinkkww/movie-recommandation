package com.service.grpc.Adapter;

import io.grpc.redisData.TotalUserDf;

import java.util.Map;

public class TotalUserDfAdapter implements GrpcAdapter {
    @Override
    public Map.Entry<byte[], Object> getInstance(String[] items) {
        TotalUserDf.Builder builder = TotalUserDf.newBuilder();
        builder.setUserId(Integer.valueOf(items[0]));
        builder.setWar(Double.valueOf(items[1]));
        builder.setAnimation(Double.valueOf(items[2]));
        builder.setHorror(Double.valueOf(items[3]));
        builder.setSciFi(Double.valueOf(items[4]));
        builder.setFantasy(Double.valueOf(items[5]));
        builder.setThriller(Double.valueOf(items[6]));
        builder.setCrime(Double.valueOf(items[7]));
        builder.setMystery(Double.valueOf(items[8]));
        builder.setDocumentary(Double.valueOf(items[9]));
        builder.setChildren(Double.valueOf(items[10]));
        builder.setAction(Double.valueOf(items[11]));
        builder.setAdventure(Double.valueOf(items[12]));
        builder.setMusical(Double.valueOf(items[13]));
        builder.setFilmNoir(Double.valueOf(items[14]));
        builder.setDrama(Double.valueOf(items[15]));
        builder.setRomance(Double.valueOf(items[16]));
        builder.setComedy(Double.valueOf(items[17]));
        builder.setWestern(Double.valueOf(items[18]));
        builder.setNone(Double.valueOf(items[19]));
        TotalUserDf instance = builder.build();
        return Map.entry(("user_like_" + items[0]).getBytes(), instance);
    }
}

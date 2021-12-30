package com.service.grpc.Adapter;

import io.grpc.redisData.MovieModReply;

import java.util.Arrays;
import java.util.Map;

public class MovieModAdapter implements GrpcAdapter {

    @Override
    public Map.Entry<byte[], Object> getInstance(String[] items) {
        MovieModReply.Builder builder = MovieModReply.newBuilder();
        try {
            builder.setMovieId(Integer.valueOf(items[0]));
            builder.setThriller(Integer.valueOf(items[1]));
            builder.setDocumentary(Integer.valueOf(items[2]));
            builder.setWar(Integer.valueOf(items[3]));
            builder.setMusical(Integer.valueOf(items[4]));
            builder.setCrime(Integer.valueOf(items[5]));
            builder.setDrama(Integer.valueOf(items[6]));
            builder.setHorror(Integer.valueOf(items[7]));
            builder.setAdventure(Integer.valueOf(items[8]));
            builder.setChildren(Integer.valueOf(items[9]));
            builder.setSciFi(Integer.valueOf(items[10]));
            builder.setComedy(Integer.valueOf(items[11]));
            builder.setMystery(Integer.valueOf(items[12]));
            builder.setWestern(Integer.valueOf(items[13]));
            builder.setFilmNoir(Integer.valueOf(items[14]));
            builder.setFantasy(Integer.valueOf(items[15]));
            builder.setAnimation(Integer.valueOf(items[16]));
            builder.setAction(Integer.valueOf(items[17]));
            builder.setRomance(Integer.valueOf(items[18]));
            builder.setNone(Integer.valueOf(items[19]));
        } catch (Exception e) {
            System.out.println(Arrays.toString(items));
        }
        MovieModReply instance = builder.build();
        String key = "movie_" + instance.getMovieId();
        return Map.entry(key.getBytes(), instance);
    }
}

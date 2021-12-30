package com.service.grpc.redisData;

import com.google.protobuf.Descriptors;
import com.google.protobuf.GeneratedMessageV3;
import com.service.grpc.Adapter.GrpcAdapter;
import io.grpc.redisData.MovieModReply;
import io.grpc.redisData.MovieReply;
import io.grpc.redisData.TotalUserDf;
import io.grpc.redisData.UserRecommendReply;
import redis.clients.jedis.Jedis;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author chenbo
 */
public class ProtoUtils {
    public final static int MOVIES_MOD = 1;
    public final static int TOTAL_USER_DISLIKE_DF = 2;
    public final static int TOTAL_USER_LIKE_DF = 3;
    public final static int USER_RECOMMEND = 4;
    public final static int ITEM_CF_RESULT = 5;
    public final static int USER_CF_RESULT = 6;
    public static final Jedis jedis = new Jedis("localhost");

    /**
     * fileName is to define where the data is
     * adapter is an abstract class to define how to deal the data from the file
     *
     * @param fileName
     * @param adapter
     */
    public static void sendBinaryString(String fileName, GrpcAdapter adapter) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            reader.readLine();
            String line = null;
            int pop = 0;
            while ((line = reader.readLine()) != null) {
                try {
                    Map.Entry<byte[], Object> instance = adapter.getInstance(line.split(","));
                    jedis.set(instance.getKey(), serialize(instance.getValue()));
                } catch (Exception e) {
                    pop++;
                }
            }
            System.out.println(pop);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * serialize an object to byte[] to store it in the redis
     *
     * @param object
     * @return byte[]
     */
    private static byte[] serialize(Object object) {
        ObjectOutputStream oos = null;
        ByteArrayOutputStream baos = null;
        try {
            baos = new ByteArrayOutputStream();
            oos = new ObjectOutputStream(baos);
            oos.writeObject(object);
            byte[] bytes = baos.toByteArray();
            return bytes;
        } catch (Exception e) {
        }
        return null;
    }

    /**
     * get a object from the redis
     *
     * @param id
     * @return null or an Object
     */
    public static Object getObject(int id, int getMethod) {
        ByteArrayInputStream stream = null;
        Object reply = null;
        String key = "";

        try {
            if (getMethod == MOVIES_MOD) {
                key = "movie_" + id;
            } else if (getMethod == TOTAL_USER_DISLIKE_DF) {
                key = "user_dislike_" + id;
            } else if (getMethod == TOTAL_USER_LIKE_DF) {
                key = "user_like_" + id;
            } else if (getMethod == USER_RECOMMEND) {
                key = "recall_" + id;
            } else if (getMethod == ITEM_CF_RESULT) {
                key = "item_result_" + id;
            } else if (getMethod == USER_CF_RESULT) {
                key = "user_result_" + id;
            }
            stream = new ByteArrayInputStream(jedis.get(key.getBytes()));
        } catch (Exception e) {
            return null;
        }


        try {
            ObjectInputStream inputStream = new ObjectInputStream(new BufferedInputStream(stream));
            try {
                reply = inputStream.readObject();
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }
        } catch (
                IOException e) {
            e.printStackTrace();
        }
        return reply;

    }

    private static String getMovieModString(UserRecommendReply recommend) {
        List<Object> list = getProtoClassValues(recommend);
        list.remove(0);
        String movieMod = "";
        for (Object o : list) {
            MovieModReply movieModReply = (MovieModReply) getObject((Integer) o, MOVIES_MOD);
            String part = "";
            part = part + movieModReply.getWar() + ",";
            part = part + movieModReply.getAnimation() + ",";
            part = part + movieModReply.getHorror() + ",";
            part = part + movieModReply.getSciFi() + ",";
            part = part + movieModReply.getFantasy() + ",";
            part = part + movieModReply.getThriller() + ",";
            part = part + movieModReply.getCrime() + ",";
            part = part + movieModReply.getMystery() + ",";
            part = part + movieModReply.getDocumentary() + ",";
            part = part + movieModReply.getChildren() + ",";
            part = part + movieModReply.getAction() + ",";
            part = part + movieModReply.getAdventure() + ",";
            part = part + movieModReply.getMusical() + ",";
            part = part + movieModReply.getFilmNoir() + ",";
            part = part + movieModReply.getDrama() + ",";
            part = part + movieModReply.getRomance() + ",";
            part = part + movieModReply.getComedy() + ",";
            part = part + movieModReply.getWestern() + ",";
            part = part + movieModReply.getNone() + "\n";
            movieMod = movieMod + part;
        }
        return movieMod;
    }

    private static String getMovieId(UserRecommendReply recommend) {
        List<Object> list = getProtoClassValues(recommend);
        list.remove(0);
        String s = "";
        for (Object o : list) {
            s = s + String.valueOf(o) + ",";
        }
        return s;
    }

    private static List<Object> getProtoClassValues(GeneratedMessageV3 message) {
        List<Object> list = new ArrayList<>();
        for (Map.Entry<Descriptors.FieldDescriptor, Object> fieldDescriptorObjectEntry : message.getAllFields().entrySet()) {
            list.add(fieldDescriptorObjectEntry.getValue());
        }
        return list;
    }

    private static String getLikeOrDislike(TotalUserDf userDf) {
        String result = "";

        for (int i = 0; i < 30; i++) {
            result = result + userDf.getWar() + ",";
            result = result + userDf.getAnimation() + ",";
            result = result + userDf.getHorror() + ",";
            result = result + userDf.getSciFi() + ",";
            result = result + userDf.getFantasy() + ",";
            result = result + userDf.getThriller() + ",";
            result = result + userDf.getCrime() + ",";
            result = result + userDf.getMystery() + ",";
            result = result + userDf.getDocumentary() + ",";
            result = result + userDf.getChildren() + ",";
            result = result + userDf.getAction() + ",";
            result = result + userDf.getAdventure() + ",";
            result = result + userDf.getMusical() + ",";
            result = result + userDf.getFilmNoir() + ",";
            result = result + userDf.getDrama() + ",";
            result = result + userDf.getRomance() + ",";
            result = result + userDf.getComedy() + ",";
            result = result + userDf.getWestern() + ",";
            result = result + userDf.getNone() + "\n";
        }
        return result;
    }

    public static MovieReply getMessage(int userId) {
        MovieReply.Builder builder = MovieReply.newBuilder();
        //movieList
        String movieList = "";

        UserRecommendReply recommend = (UserRecommendReply) getObject(userId, USER_RECOMMEND);
        movieList = movieList + getMovieId(recommend);


        UserRecommendReply itemRecommend = (UserRecommendReply) getObject(userId, ITEM_CF_RESULT);

        movieList = movieList + getMovieId(itemRecommend);


        UserRecommendReply userCFRecommend = (UserRecommendReply) getObject(userId, USER_CF_RESULT);
        movieList = movieList + getMovieId(userCFRecommend);
        movieList = String.copyValueOf(movieList.toCharArray(), 0, movieList.length() - 1) + "\n";
        builder.setMovieList(movieList);

        //userDislike
        String userDislike = "";
        TotalUserDf dislikeDf = (TotalUserDf) getObject(userId, TOTAL_USER_DISLIKE_DF);
        userDislike = getLikeOrDislike(dislikeDf);
        builder.setUserDislike(userDislike);


        String userLike = "";
        TotalUserDf likeDf = (TotalUserDf) getObject(userId, TOTAL_USER_LIKE_DF);
        userLike = getLikeOrDislike(likeDf);
        builder.setUserLike(userLike);


        String movieMod = "";

        movieMod = movieMod + getMovieModString(recommend);
        movieMod = movieMod + getMovieModString(itemRecommend);
        movieMod = movieMod + getMovieModString(userCFRecommend);

        builder.setMovieMod(movieMod);
        return builder.build();
    }
}

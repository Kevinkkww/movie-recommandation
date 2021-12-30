package com.service.grpc.redisData;

import com.google.protobuf.Descriptors;
import com.service.grpc.Adapter.UserRecommendAdapter;
import io.grpc.redisData.MovieReply;
import io.grpc.redisData.UserRecommendReply;

import java.io.*;
import java.util.HashSet;
import java.util.Map;

public class Test {
    private static void test1() throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/recall.csv"));
        String line;
        int poll = 0;
        BufferedWriter writer = new BufferedWriter(new FileWriter("src/main/resources/pollUser.csv"));
        while ((line = reader.readLine()) != null) {
            try {
                String[] split = line.split(",");
                MovieReply message = ProtoUtils.getMessage(Integer.valueOf(split[0]));
                if (message == null) {
                    writer.write(split[0] + "\n");
                }
            } catch (Exception e) {
                System.out.println(line.split(",")[0] + "   发生了异常");
            }
        }
        writer.close();
        reader.close();
    }

    private static HashSet<Integer> getUserIdHash(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String s;
        HashSet<Integer> hashSet = new HashSet<>();
        while ((s = reader.readLine()) != null) {
            String[] split = s.split(",");
            int userId = Integer.valueOf(split[0]);
            if (!hashSet.contains(userId)) {
                hashSet.add(userId);
            }
        }
        return hashSet;
    }

    private static void test2() {
        try {
            int difference = 0;
            HashSet<Integer> itemHash = getUserIdHash("src/main/resources/recall.csv");
            HashSet<Integer> userHash = getUserIdHash("src/main/resources/UserCFresult.csv");
            for (Integer hash : itemHash) {
                if (!userHash.contains(hash)) {
                    difference++;
                }
            }
            System.out.println(difference);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            test1();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}

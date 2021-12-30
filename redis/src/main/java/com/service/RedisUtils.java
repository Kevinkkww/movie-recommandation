package com.service;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;


public class RedisUtils {
    public static void main(String[] args) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/movies_mod.csv"));
            BufferedWriter writer = new BufferedWriter(new FileWriter("src/main/resources/movieId.csv"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] split = line.split(",");
                writer.write(split[0]+"\n");
            }
            writer.close();
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

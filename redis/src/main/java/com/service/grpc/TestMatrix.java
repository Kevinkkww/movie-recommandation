package com.service.grpc;
import com.service.grpc.redisData.ProtoUtils;
import io.grpc.redisData.MovieModReply;
import io.grpc.redisData.TotalUserDf;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import static java.lang.Math.exp;


public class TestMatrix {
    public static void main(String[] args) {
        try {
            recall();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static double sigmoid(double x) {
        if (x >= 0) {
            return 1.0 / (1 + exp(-x));
        } else {
            return exp(x) / (1 + exp(x));
        }
    }

    public static double getPrediction(Matrix matrix, double w_0, Matrix w, Matrix v) {
        double result = 0;

        Matrix inter_1, inter_2;

        inter_1 = matrix.mtimes(v);
        inter_2 = matrix.times(matrix).mtimes(v.times(v));
        double interaction = inter_1.times(inter_1).minus(inter_2).divide(2).getAbsoluteValueSum();
        double p = w_0 + matrix.mtimes(w).norm1() + interaction;
        result = sigmoid(p);

        return result;
    }

    public static void recall() throws IOException {
        Matrix v, w;
        double[] a = new double[]{0.15783855,
                0.03310976,
                -0.34660972,
                -0.09044366,
                -0.01485935,
                -0.33597882,
                0.16979688,
                0.07520457,
                0.01571355,
                -0.28076161,
                -0.18315379,
                0.07314125,
                -0.14893313,
                0.11648317,
                0.00276905,
                -0.09590032,
                -0.48833601,
                -0.13191399,
                0.,
                -0.13696259,
                0.,
                -0.07815073,
                -0.13623405,
                -0.05887868,
                -0.12201653,
                0.,
                -0.33225543,
                -0.20956276,
                0.,
                0.10594806,
                -0.03265493,
                0.0316002,
                0.04757184,
                -0.01667327,
                -0.7846659,
                0.10573709,
                0.12134434,
                -0.02499259,
                -0.01211823};
        w = DenseMatrix.Factory.importFromArray(a);

        double[] b = new double[]{-0.2259373,
                -0.18396271,
                -0.20002889,
                -0.07175112,
                -0.07754284,
                -0.20903879,
                -0.3389806,
                -0.0803205,
                -0.22625079,
                -0.09243298,
                -0.02424713,
                -0.0319998,
                -0.18168655,
                -0.2470192,
                -0.3682893,
                -0.11219277,
                -0.50205985,
                -0.11055543,
                -0.21869848,
                -0.28122243,
                -0.21869848,
                -0.18335952,
                -0.15091583,
                -0.20984153,
                -0.15808447,
                -0.21869848,
                -0.06692586,
                -0.10824962,
                -0.21869848,
                -0.30450257,
                -0.25072977,
                -0.26551378,
                -0.26476858,
                -0.21301282,
                0.14313395,
                -0.33210679,
                -0.32264859,
                -0.20999349,
                -0.21177488};
        v = DenseMatrix.Factory.importFromArray(b);

        double w_0 = -1.430884713434919;


        for (int user_id = 1; user_id <= 162542; user_id++) {
            int[] result = new int[10];
            int total = 0;
            double[] features = new double[39];
            TotalUserDf user = (TotalUserDf) ProtoUtils.getObject(user_id, 3);
            if (user == null) {
                continue;
            }
            features[20] = user.getWar();
            features[21] = user.getAnimation();
            features[22] = user.getHorror();
            features[23] = user.getSciFi();
            features[24] = user.getFantasy();
            features[25] = user.getThriller();
            features[26] = user.getCrime();
            features[27] = user.getMystery();
            features[28] = user.getDocumentary();
            features[29] = user.getChildren();
            features[30] = user.getAction();
            features[31] = user.getAdventure();
            features[32] = user.getMusical();
            features[33] = user.getFilmNoir();
            features[34] = user.getDrama();
            features[35] = user.getRomance();
            features[36] = user.getComedy();
            features[37] = user.getWestern();
            features[38] = user.getNone();

            for (int i = 0; i < 1500; i++) {
                if (total == 10) {
                    break;
                }
                int movie_id = (int) (1 + Math.random() * (189901 - 1 + 1));

                MovieModReply movie = (MovieModReply) ProtoUtils.getObject(movie_id, 1);
                if (movie == null) {
                    i -= 1;
                    continue;

                }

                features[0] = movie.getWar();
                features[1] = movie.getAnimation();
                features[2] = movie.getHorror();
                features[3] = movie.getSciFi();
                features[4] = movie.getFantasy();
                features[5] = movie.getThriller();
                features[6] = movie.getCrime();
                features[7] = movie.getMystery();
                features[8] = movie.getDocumentary();
                features[9] = movie.getChildren();
                features[10] = movie.getAction();
                features[11] = movie.getAdventure();
                features[12] = movie.getMusical();
                features[13] = movie.getFilmNoir();
                features[14] = movie.getDrama();
                features[15] = movie.getRomance();
                features[16] = movie.getComedy();
                features[17] = movie.getWestern();
                features[18] = movie.getNone();
//                System.out.println(getDoubleArrayString(features));
                Matrix matrix = DenseMatrix.Factory.importFromArray(features);

                double prediction = getPrediction(matrix, w_0, w.transpose(), v.transpose());

                if (prediction > 0.5) {
                    result[total] = movie_id;
                    total += 1;
                }
            }
            if (total < 10) {
                for (; total < 10; total++) {
                    int s=(int) (1 + Math.random() * (189901 - 1 + 1));
                    MovieModReply movie = (MovieModReply) ProtoUtils.getObject(s, 1);
                    if (movie == null) {
                        total -= 1;
                        continue;
                    }
                    result[total] = s;
                    System.out.println(total);
                }
            }
            BufferedWriter errorWriter = new BufferedWriter(new FileWriter("/Users/lichenbo/Desktop/recall.csv", true));
            errorWriter.write(user_id + "," + getArrayString(result) + "\n");
            errorWriter.close();
        }
    }

    public static String getArrayString(int[] s) {
        String result = "";
        for (int i : s) {
            result = result + i + ",";
        }
        return result;
    }

    public static String getDoubleArrayString(double[] s) {
        String result = "";
        for (double i : s) {
            result = result + i + ",";
        }
        return result;
    }
}

package main.java.experiments;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.DataSet;
import es.upm.etsisi.cf4j.data.RandomSplitDataSet;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BNMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.BiasedMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.NMF;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import es.upm.etsisi.cf4j.util.Range;

public class HyperparametersOptimization {

    private static final String BINARY_FILE = "datasets/ml100k.dat";
    private static final long SEED = 1337;
//    private static final String BINARY_FILE = "datasets/filmtrust.cf4j";

    private static int NUM_ITERS = 100;


    public static void main(String[] args) {
        DataSet ml100k = new RandomSplitDataSet(BINARY_FILE, 0.2, 0.2, "::", SEED);
        DataModel dataModel = new DataModel(ml100k);


        // Test PMF model

        double minError = Double.MAX_VALUE;
        String best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {
            for (double lambda : Range.ofDoubles(0.005, 0.005, 20)) {
                for (double gamma : Range.ofDoubles(0.005, 0.005, 20)) {

                    Recommender pmf = new PMF(dataModel, numTopics, NUM_ITERS, lambda, gamma, SEED);
                    pmf.fit();
                    QualityMeasure mae = new MAE(pmf);
                    double mae_score = mae.getScore();

                    System.out.println("numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae_score);

                    if (mae_score < minError) {
                        minError = mae_score;
                        best = "numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae_score;
                    }
                }
            }
        }

        System.out.println("\nBest result for PMF => " + best);


        // Test BiasedMF model

        minError = Double.MAX_VALUE;
        best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {
            for (double lambda : Range.ofDoubles(0.005, 0.005, 20)) {
                for (double gamma : Range.ofDoubles(0.005, 0.005, 20)) {

                    Recommender bmf = new BiasedMF(dataModel, numTopics, NUM_ITERS, lambda, gamma, SEED);
                    bmf.fit();
                    QualityMeasure mae = new MAE(bmf);
                    double mae_score = mae.getScore();

                    System.out.println("numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae_score);

                    if (mae_score < minError) {
                        minError = mae_score;
                        best = "numTopics = " + numTopics + "; lambda = " + lambda + "; gamma = " + gamma + "; mae = " + mae_score;
                    }
                }
            }
        }

        System.out.println("\nBest result for BiasedMF => " + best);


        // Test NMF model

        minError = Double.MAX_VALUE;
        best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {

            Recommender nmf = new NMF(dataModel, numTopics, NUM_ITERS, SEED);
            nmf.fit();
            QualityMeasure mae = new MAE(nmf);
            double mae_score = mae.getScore();

            System.out.println("numTopics = " + numTopics + "; mae = " + mae_score);

            if (mae_score < minError) {
                minError = mae_score;
                best = "numTopics = " + numTopics + "; mae = " + mae_score;
            }
        }

        System.out.println("\nBest result for NMF => " + best);


        // Test BNMF model

        minError = Double.MAX_VALUE;
        best = "";

        for (int numTopics : Range.ofIntegers(4,2,5)) {
            for (double alpha : Range.ofDoubles(0.1, 0.1, 9)) {
                for (double beta : Range.ofDoubles(5, 5, 5)) {

                    Recommender bnmf = new BNMF(dataModel, numTopics, NUM_ITERS, alpha, beta, SEED);
                    bnmf.fit();
                    QualityMeasure mae = new MAE(bnmf);
                    double mae_score = mae.getScore();

                    System.out.println("numTopics = " + numTopics + "; alpha = " + alpha + "; beta = " + beta + "; mae = " + mae_score);

                    if (mae_score < minError) {
                        minError = mae_score;
                        best = "numTopics = " + numTopics + "; alpha = " + alpha + "; beta = " + beta + "; mae = " + mae_score;
                    }
                }
            }
        }

        System.out.println("\nBest result for BNMF => " + best);
    }
}
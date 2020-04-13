package experiments;

import com.github.ferortega.cf4j.data.DataModel;
import com.github.ferortega.cf4j.data.DataSet;
import com.github.ferortega.cf4j.data.RandomSplitDataSet;
import com.github.ferortega.cf4j.qualityMeasure.QualityMeasure;
import com.github.ferortega.cf4j.qualityMeasure.prediction.MAE;
import com.github.ferortega.cf4j.qualityMeasure.prediction.MSE;
import com.github.ferortega.cf4j.recommender.Recommender;
import com.github.ferortega.cf4j.recommender.matrixFactorization.BNMF;
import com.github.ferortega.cf4j.recommender.matrixFactorization.BiasedMF;
import com.github.ferortega.cf4j.recommender.matrixFactorization.NMF;
import com.github.ferortega.cf4j.recommender.matrixFactorization.PMF;
import mf.EMF;

import java.io.IOException;

public class BaselinesComparison {

    private static long SEED = 1337;
    private static int NUM_ITERS = 100;

    private static final String BINARY_FILE = "datasets/ml100k.dat";

    private static int PMF_NUM_TOPICS = 6;
    private static double PMF_LAMBDA = 0.085;
    private static double PMF_GAMMA = 0.01;

    private static int BIASED_MF_NUM_TOPICS = 6;
    private static double BIASED_MF_LAMBDA = 0.06;
    private static double BIASED_MF_GAMMA = 0.01;

    private static int NMF_NUM_TOPICS = 6;

    private static int BNMF_NUM_TOPICS = 6;
    private static double BNMF_ALPHA = 0.9;
    private static int BNMF_BETA = 5;

    private static int EMF_NUM_TOPICS = 6;
    private static double EMF_LEARNING_RATE = 0.001;
    private static double EMF_REGULARIZARION = 0.095;

    private static String [] EMF_FUNCS = {
            "* - cos cos log exp atan pu4 - atan -- qi3 exp -- atan log exp pu1 exp cos log cos log exp - exp cos pu2 exp -- atan log atan sin cos cos atan atan exp - log exp atan cos log exp atan pu4 pu2",
            "- + + exp sin qi3 exp + atan pu2 cos One pu1 * + pu1 sin + pu1 sin + + pu1 + + exp qi5 exp + * pu1 pu0 sin + + pu1 * + * * One One pu0 pu1 qi2 One + pu1 * + * * One One qi4 pu1 qi2 One qi2",
            "- - inv inv + inv inv + sin One cos Zero - inv inv + inv inv + sin atan -- + + * qi0 pu4 pu0 * qi0 pu4 cos Zero cos Zero + * inv inv qi3 pu4 -- sin atan + + qi0 pu0 * qi0 qi0 + * inv inv qi3 pu4 -- sin atan + + qi0 pu0 * cos Zero pu4 atan -- + + qi0 pu0 * inv + inv inv + sin One cos Zero cos Zero pu4",
            "-- + - - qi0 + + + One One * pu3 qi0 One + One pu2 * pu0 qi0",
            "-- -- + + atan qi1 + + atan pu5 exp pu3 atan + -- - exp qi2 exp atan pu4 qi5 exp qi1",
            "log exp + exp atan exp atan - exp atan - exp pu0 qi5 qi5 exp atan - pu3 qi5",
            "* - -- -- cos -- -- pu1 atan inv -- qi3 exp cos * atan -- exp atan pu1 -- qi4",
            "- + - qi4 pu2 inv * cos sin - qi4 pu2 atan exp qi2 - cos - qi4 pu4 cos cos * - - pu2 * qi4 One pu2 -- atan pu0",
            "+ exp qi0 - exp atan pu0 + - Zero - exp qi0 qi5 + * pu5 qi5 - pu4 qi2",
            "exp atan pow pow * exp exp pu5 inv exp qi0 exp pu4 exp inv exp pu5"
    };

//    private static final String BINARY_FILE = "datasets/filmtrust.cf4j";
//
//    private static int PMF_NUM_TOPICS = 6;
//    private static double PMF_LAMBDA = 0.085;
//    private static double PMF_GAMMA = 0.01;
//
//    private static int BIASED_MF_NUM_TOPICS = 12;
//    private static double BIASED_MF_LAMBDA = 0.095;
//    private static double BIASED_MF_GAMMA = 0.03;
//
//    private static int NMF_NUM_TOPICS = 4;
//
//    private static int BNMF_NUM_TOPICS = 4;
//    private static double BNMF_ALPHA = 0.8;
//    private static int BNMF_BETA = 5;
//
//    private static int EMF_NUM_TOPICS = 10;
//    private static double EMF_LEARNING_RATE = 0.0035;
//    private static double EMF_REGULARIZARION = 0.095;
//
//    private static String [] EMF_FUNCS = {
//            "exp atan + cos pu5 exp + + pu3 qi2 pu3",
//            "-- -- + sin -- -- pu0 + One exp sin cos exp + pu6 -- + sin pu9 + sin qi4 exp sin -- + One exp sin cos -- + sin sin pu9 + One exp sin cos exp + pu6 -- + sin qi4 exp sin -- sin pu9",
//            "inv exp atan - pu2 exp exp atan - exp - exp - pu2 exp atan - pu2 pu7 exp - - pu7 qi3 exp atan - pu2 exp - pu2 exp atan - pu2 exp exp atan - exp - pu7 exp - - pu2 exp exp atan - exp - pu2 exp - - pu7 qi3 exp atan - pu2 exp - pu2 exp atan - pu2 exp exp - pu2 exp atan - pu2 - pu2 pu7 - pu7 pu2 pu7 - pu7 pu2 - pu7 pu2",
//            "+ pu4 exp cos qi6",
//            "exp atan - exp exp pu1 -- qi9",
//            "exp atan + qi4 + + pu3 atan atan pu2 inv pu2",
//            "+ atan exp qi2 exp cos pu2",
//            "+ exp pu3 exp cos sin exp -- atan qi0",
//            "exp inv cos cos exp cos exp atan - * One exp * qi6 cos cos exp * pu7 qi2 exp atan - * One exp * qi6 cos pu4 pu0",
//            "exp atan + * atan + + atan + * atan + + qi5 atan + qi5 atan - inv pu8 * qi5 inv inv pu8 inv pu8 + qi5 atan - inv pu8 * qi5 inv inv pu8 inv pu8 atan + qi5 atan - inv pu8 * qi5 inv + qi5 atan qi5 inv pu8 + qi5 atan - inv pu8 * qi5 inv inv pu8 inv pu8"
//    };

    public static void main(String[] args) {

        // define series

        String [] series = new String [4 + EMF_FUNCS.length];

        series[0] = "PMF";
        series[1] = "BiasedMF";
        series[2] = "NMF";
        series[3] = "BNMF";

        for (int i = 1; i <= EMF_FUNCS.length; i++) {
            series[3 + i] = "EMF_" + i;
        }

        // define quality measures

        double [] mae_vector = new double [series.length];
        double [] mse_vector = new double [series.length];

        // load dataset

        DataSet ml100k = null;
        try {
            ml100k = new RandomSplitDataSet(BINARY_FILE, 0.2f, 0.2f, "::", SEED);
        } catch (IOException e) {
            System.out.println("There was an error when loading file " + BINARY_FILE);
            e.printStackTrace();
            System.exit(-1);
        }
        DataModel dataModel = new DataModel(ml100k);


        // test series

        for (int s = 0; s < series.length; s++) {
            String serie = series[s];

            Recommender fm;


            switch (serie) {
                case "PMF":
                    fm = new PMF(dataModel, PMF_NUM_TOPICS, NUM_ITERS, PMF_LAMBDA, PMF_GAMMA, SEED);

                    break;
                case "BiasedMF":
                    fm = new BiasedMF(dataModel, BIASED_MF_NUM_TOPICS, NUM_ITERS, BIASED_MF_LAMBDA, BIASED_MF_GAMMA, SEED);

                    break;
                case "NMF":
                    fm = new NMF(dataModel, NMF_NUM_TOPICS, NUM_ITERS, SEED);

                    break;
                case "BNMF":
                    fm = new BNMF(dataModel, BNMF_NUM_TOPICS, NUM_ITERS, BNMF_ALPHA, BNMF_BETA, SEED);

                    break;
                default:  // serie.equals("EMF_<id>")
                    int index = Integer.parseInt(serie.split("_")[1]) - 1;
                    String func = EMF_FUNCS[index];
                    fm = new EMF(func, dataModel, EMF_NUM_TOPICS, NUM_ITERS, EMF_REGULARIZARION, EMF_LEARNING_RATE, SEED,true);
                    break;
            }

            fm.fit();
            QualityMeasure mae = new MAE(fm);
            QualityMeasure mse = new MSE(fm);

            mae_vector[s] = mae.getScore();
            mse_vector[s] = mse.getScore();


            // print results

            System.out.println("\nMethod;MAE;MSE");
            for (int i = 0; i < series.length; i++) {
                System.out.println(series[i] + ";" + mae_vector[i] + ";" + mse_vector[i]);
            }
        }
    }
}

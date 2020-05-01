package experiments;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.DataSet;
import es.upm.etsisi.cf4j.data.RandomSplitDataSet;
import mf.AlgorithmProgress;
import mf.MatrixFactorizationProblem;
import org.moeaframework.Executor;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.PRNG;
import org.moeaframework.core.Solution;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class OneShot {
    public static void main(String[] args) throws IOException {
        // Datamodel
        DataSet ml100k = null;
        try {
            ml100k = new RandomSplitDataSet("datasets/ml100k.dat", 0.2f, 0.2f, "::");
        } catch (IOException e) {
            System.out.println("There was an error when loading file " + "datasets/ml100k.dat");
            e.printStackTrace();
            System.exit(-1);
        }
        DataModel model = new DataModel(ml100k);

//        MatrixFactorizationProblem(DataModel model, int numFactors, int iters, double regularization, double learningRate, int nRecommendations)

        // Statistics and indicators
//        Instrumenter instrumenter = new Instrumenter()
//                .withProblemClass(MatrixFactorizationProblem.class, model, 6, 100, 0.055, 0.0001, 10)
//                .withFrequency(100)
//                .withReferenceSet(new File("pf_ml100k"))
//                .attachHypervolumeCollector()
//                .attachInvertedGenerationalDistanceCollector()
//                .attachAdditiveEpsilonIndicatorCollector()
//                .attachSpacingCollector()
//                .attachElapsedTimeCollector()
//                .attachPopulationSizeCollector();

        PRNG.setSeed(19841024);

        NondominatedPopulation results = new Executor()
                .withAlgorithm("NSGAII")
                .withProblemClass(MatrixFactorizationProblem.class, model, 6, 100, 0.055, 0.0001, 10)
                .withProperty("populationSize", 50)
                .withProperty("operator", "bx+ptm")
                .withProperty("bx.probability", 0.7)
                .withProperty("ptm.probability", 0.2)
                .withMaxEvaluations(10000)
//                .checkpointEveryIteration()
//                .withCheckpointFile(new File("ml100k_6_5e-2_1e-4_10.chkp"))
//                .withInstrumenter(instrumenter)
                .withProgressListener(new AlgorithmProgress())
                .distributeOnAllCores()
                .run();

//        Accumulator acc = instrumenter.getLastAccumulator();
//        acc.saveCSV(new File("ml100k_6_5e-2_1e-4_10.csv"));

        // Output file with unique filename
        SimpleDateFormat df = new SimpleDateFormat("yyyyMMddhhmmssSSS");
        Date d = new Date();
        FileWriter fileWriter = new FileWriter(df.format(d) + "_pf_ml100k_6_5e-2_1e-4_10");
        for (Solution solution : results) {
            fileWriter.write(MatrixFactorizationProblem.translate(solution.getVariable(0).toString()) + "," + solution.getObjective(0) + "," + solution.getObjective(1) + "," + solution.getObjective(2)+"\n");
        }
        fileWriter.close();
    }
}

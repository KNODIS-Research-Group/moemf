package experiments;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.qualityMeasure.recommendation.Diversity;
import es.upm.etsisi.cf4j.qualityMeasure.recommendation.Novelty;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.recommender.matrixFactorization.PMF;
import es.upm.etsisi.cf4j.util.Range;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.stream.StreamSupport;

public class BaselinesParetoFrontier {

    public static void main (String[] args) throws Exception {

        DataModel datamodel = BenchmarkDataModels.MovieLens100K();

        FileWriter out = new FileWriter("pmf_pareto_frontier.csv");
        CSVPrinter printer = new CSVPrinter(out, CSVFormat.DEFAULT.withHeader("params", "mae", "-novelty", "diversity"));

        ParamsGrid paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("lambda", Range.ofDoubles(0.005, 0.005, 20));
        paramsGrid.addParam("gamma", Range.ofDoubles(0.005, 0.005, 20));

        paramsGrid.addFixedParam("numIters", 50);
        paramsGrid.addFixedParam("seed", 4815162342L);

        Iterator<Map<String, Object>> iter = paramsGrid.getDevelopmentSetIterator();

        while (iter.hasNext()) {
            Map<String, Object> params = iter.next();

            System.out.println("Evaluating " + params.toString());

            Recommender recommender = new PMF(datamodel, params);
            recommender.fit();

            QualityMeasure mae = new MAE(recommender);
            double err = mae.getScore();

            QualityMeasure novelty = new Novelty(recommender, 10);
            double nov = novelty.getScore();

            QualityMeasure diversity = new Diversity(recommender, 10);
            double div = diversity.getScore();

            printer.printRecord(params.toString(), err, -nov, div);
        }

        printer.close();
        out.close();
    }
}

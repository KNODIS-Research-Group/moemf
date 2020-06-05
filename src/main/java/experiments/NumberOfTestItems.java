package experiments;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.recommender.knn.itemSimilarityMetric.Cosine;
import es.upm.etsisi.cf4j.util.plot.HistogramPlot;
import es.upm.etsisi.cf4j.util.process.Parallelizer;

import java.io.IOException;

public class NumberOfTestItems {

    public static void main (String[] args) throws IOException {

        DataModel datamodel = BenchmarkDataModels.MovieLens100K();

        HistogramPlot histogram = new HistogramPlot("number of test items", 11);

        for (TestUser testUser : datamodel.getTestUsers()) {
            histogram.addValue(testUser.getNumberOfTestRatings());
        }

        histogram.draw();
    }
}

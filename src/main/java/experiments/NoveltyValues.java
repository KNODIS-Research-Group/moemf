package experiments;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.recommender.knn.itemSimilarityMetric.Cosine;
import es.upm.etsisi.cf4j.util.Maths;
import es.upm.etsisi.cf4j.util.plot.HistogramPlot;
import es.upm.etsisi.cf4j.util.process.Parallelizer;

import java.io.IOException;

public class NoveltyValues {

    public static void main (String[] args) throws IOException {

        DataModel datamodel = BenchmarkDataModels.MovieLens100K();

        HistogramPlot histogram = new HistogramPlot("item novelty", 11);

        for (Item item : datamodel.getItems()) {
            double pi = (double) item.getNumberOfRatings() / (double) datamodel.getNumberOfRatings();
            double novelty = -Maths.log(pi, 2);
            histogram.addValue(novelty);
        }

        histogram.draw();
    }
}

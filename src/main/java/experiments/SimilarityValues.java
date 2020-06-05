package experiments;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.recommender.knn.itemSimilarityMetric.Cosine;
import es.upm.etsisi.cf4j.util.plot.HistogramPlot;
import es.upm.etsisi.cf4j.util.process.Parallelizer;

import java.io.IOException;

public class SimilarityValues {

    public static void main (String[] args) throws IOException {

        DataModel datamodel = BenchmarkDataModels.MovieLens100K();

        Cosine cosine = new Cosine();
        cosine.setDatamodel(datamodel);

        Parallelizer.exec(datamodel.getItems(), cosine);

        HistogramPlot histogram = new HistogramPlot("cosine similarity", 16);

        for (int i = 0; i < datamodel.getNumberOfItems(); i++) {
            double[] similarities = cosine.getSimilarities(i);
            for (int j = i+1; j < datamodel.getNumberOfItems(); j++) {
                if (!Double.isInfinite(similarities[j])) {
                    histogram.addValue(similarities[j]);
                }
            }
        }

        histogram.draw();
    }
}

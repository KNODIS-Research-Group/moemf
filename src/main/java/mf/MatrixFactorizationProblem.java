package mf;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.qualityMeasure.recommendation.Diversity;
import es.upm.etsisi.cf4j.qualityMeasure.recommendation.Novelty;
import es.upm.etsisi.cf4j.recommender.Recommender;
import mf.opers.Inverse;
import mf.opers.Negate;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.Program;
import org.moeaframework.problem.AbstractProblem;
import org.moeaframework.util.tree.*;

public class MatrixFactorizationProblem extends AbstractProblem {
    private final DataModel _model;
    private final int _numFactors;
    private final int _iters;
    private final double _learningRate;
    Rules rules = new Rules();
    private final double _regularization;
    private final int _nRecommendations;

    public MatrixFactorizationProblem(DataModel model,
                                      int numFactors,
                                      int iters,
                                      double regularization,
                                      double learningRate,
                                      int nRecommendations) {
        super(1, 3, 1);

        _model = model;
        _numFactors = numFactors;
        _iters = iters;
        _regularization = regularization;
        _learningRate = learningRate;
        _nRecommendations = nRecommendations;

        defineOpers(_numFactors);
    }

    private void defineOpers(int k) {
        rules.add(new Add());
        rules.add(new Subtract());
        rules.add(new Multiply());
        rules.add(new Power());
        rules.add(new Negate());
        rules.add(new Inverse());
        rules.add(new Log());
        rules.add(new Exp());
        rules.add(new Sin());
        rules.add(new Cos());
        rules.add(new Atan());
        rules.add(new Constant(0.0));
        rules.add(new Constant(1.0));

        for (int i = 0; i < k; i++) {
            rules.add(new Get(Number.class, "pu"+i));
            rules.add(new Get(Number.class, "qi"+i));
        }

        rules.setReturnType(Number.class);
    }

    @Override
    public void evaluate(Solution solution) {
        String func = translate(solution.getVariable(0).toString());

        Recommender emf = new EMF(_model, func, _numFactors, _iters, _regularization,
                _learningRate, 4815162342L,false);
        emf.fit();

        if (((EMF) emf).isValid()) {
            QualityMeasure mae = new MAE(emf);
            QualityMeasure novelty = new Novelty(emf, _nRecommendations);
            QualityMeasure diversity = new Diversity(emf, _nRecommendations);
            double error = mae.getScore();
            double nov = novelty.getScore();
            double div = diversity.getScore();

            solution.setObjectives(new double[]{error, -nov, div});
            solution.setConstraints(new double[] {0});
        } else {
            solution.setObjectives(new double[]{Double.MAX_VALUE, 0.0, Double.MAX_VALUE});
            solution.setConstraints(new double[] {0});
        }
    }

    public static String translate(String s) {
        return s.replace("("," ")
                .replace(")"," ")
                .replace(",", " ")
                .toLowerCase()
                .replace("program", "")
                .replace("add", "+")
                .replace("subtract", "-")
                .replace("multiply", "*")
                .replace("power", "pow")
                .replace("negate", "--")
                .replace("inverse", "inv")
                .replace("0.0", "Zero")
                .replace("1.0", "One");
    }

    @Override
    public Solution newSolution() {
        Solution solution = new Solution(this.numberOfVariables, this.numberOfObjectives, this.numberOfConstraints);
        solution.setVariable(0, new Program(rules));
        return solution;
    }
}

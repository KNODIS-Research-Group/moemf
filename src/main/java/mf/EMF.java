package main.java.mf;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.recommender.Recommender;
import sym_derivation.symderivation.SymFunction;

import java.util.HashMap;
import java.util.Random;

public class EMF extends Recommender {

    private double learningRate;
    private double regularization;
    private int numFactors;
    private int numIters;

    private SymFunction sf;

    private double [][] p;
    private double [][] q;

    private boolean verbose;

    public EMF(String func, DataModel dataModel, int numFactors, int numIters, double regularization, double learningRate, boolean verbose) {
        super(dataModel);

        Random rand = new Random(seed);

        // create model function
        this.sf = SymFunction.parse(func);

        // model hyper-parameters
        this.numFactors = numFactors;
        this.numIters = numIters;
        this.regularization = regularization;
        this.learningRate = learningRate;

        // users factors initialization
        this.p = new double[datamodel.getNumberOfUsers()][numFactors];
        for (int u = 0; u < datamodel.getNumberOfUsers(); u++) {
            for (int k = 0; k < numFactors; k++) {
                this.p[u][k] = rand.nextDouble();
            }
        }

        // items factors initialization
        this.q = new double [datamodel.getNumberOfItems()][numFactors];
        for (int i = 0; i < datamodel.getNumberOfItems(); i++) {
            for (int k = 0; k < numFactors; k++) {
                this.q[i][k] = rand.nextDouble();
            }
        }

        // verbose mode
        this.verbose = verbose;
    }

    @Override
    public void fit () {

        if (verbose) System.out.println("\nProcessing EMF...");

        // partial derivatives of the model function

        SymFunction [] puSfDiff = new SymFunction [this.numFactors];
        SymFunction [] qiSfDiff = new SymFunction [this.numFactors];

        for (int k = 0; k < this.numFactors; k++) {
            puSfDiff[k] = sf.diff("pu" + k);
            qiSfDiff[k] = sf.diff("qi" + k);
        }

        // repeat numIters times
        for (int iter = 1; iter <= this.numIters; iter++) {

            // compute gradient
            double[][] dp = new double[datamodel.getNumberOfUsers()][this.numFactors];
            double[][] dq = new double[datamodel.getNumberOfItems()][this.numFactors];

            for (int userIndex = 0; userIndex < datamodel.getNumberOfUsers(); userIndex++) {

                User user = datamodel.getUser(userIndex);

                for (int pos = 0; pos < user.getNumberOfRatings(); pos++) {
                    int itemIndex = user.getItemAt(pos);

                    HashMap <String, Double> params = getParams(p[userIndex], q[itemIndex]);

                    double prediction = sf.eval(params);
                    double error = user.getRatingAt(pos) - prediction;

                    for (int k = 0; k < this.numFactors; k++) {
                        dp[userIndex][k] += this.learningRate * (error * puSfDiff[k].eval(params) - this.regularization * p[userIndex][k]);
                        dq[itemIndex][k] += this.learningRate * (error * qiSfDiff[k].eval(params) - this.regularization * q[itemIndex][k]);
                    }
                }
            }

            // update users factors
            for (int userIndex = 0; userIndex < datamodel.getNumberOfUsers(); userIndex++) {
                for (int k = 0; k < this.numFactors; k++) {
                    p[userIndex][k] += dp[userIndex][k];
                }
            }

            // update items factors
            for (int itemIndex = 0; itemIndex < datamodel.getNumberOfItems(); itemIndex++) {
                for (int k = 0; k < this.numFactors; k++) {
                    q[itemIndex][k] += dq[itemIndex][k];
                }
            }

            if (verbose) {
                if ((iter % 10) == 0) System.out.print(".");
                if ((iter % 100) == 0) System.out.println(iter + " iterations");
            }
        }
    }

    @Override
    public double predict(int userIndex, int itemIndex) {
        HashMap <String, Double> params = getParams(this.p[userIndex], this.q[itemIndex]);
        return sf.eval(params);
    }

    private HashMap<String, Double> getParams (double [] pu, double [] qi) {
        HashMap <String, Double> map = new HashMap<>();
        for (int k = 0; k < this.numFactors; k++) {
            map.put("pu" + k, pu[k]);
            map.put("qi" + k, qi[k]);
        }
        return map;
    }
}

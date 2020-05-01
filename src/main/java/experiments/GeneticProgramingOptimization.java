package experiments;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.DataSet;
import es.upm.etsisi.cf4j.data.RandomSplitDataSet;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.qualityMeasure.recommendation.Diversity;
import es.upm.etsisi.cf4j.qualityMeasure.recommendation.Novelty;
import es.upm.etsisi.cf4j.recommender.Recommender;
import io.jenetics.*;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.Limits;
import io.jenetics.ext.SingleNodeCrossover;
import io.jenetics.ext.moea.MOEA;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.Vec;
import io.jenetics.ext.moea.VecFactory;
import io.jenetics.ext.rewriting.TreeRewriteRule;
import io.jenetics.ext.rewriting.TreeRewriter;
import io.jenetics.prog.MathRewriteAlterer;
import io.jenetics.prog.ProgramChromosome;
import io.jenetics.prog.ProgramGene;
import io.jenetics.prog.op.Op;
import io.jenetics.prog.op.Var;
import io.jenetics.util.ISeq;
import io.jenetics.util.IntRange;
import io.jenetics.util.RandomRegistry;
import mf.EMF;
import org.apache.commons.cli.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

@SuppressWarnings({"unchecked"})
public class GeneticProgramingOptimization {

	private static final String BINARY_FILE = "datasets/ml100k.dat";
	private static final long SEED = 1337;
    private static final int NUM_RECS = 10;
	private static final IntRange NUM_RESULTS = IntRange.of(10,20);

	private static int NUM_TOPICS = 6;
	private static double REGULARIZATION = 0.055;
	private static double LEARNING_RATE = 0.0001;
	private static int GENS = 150;
	private static double PBMUT = 0.1;
	private static double PBX = 0.5;
	private static int POP_SIZE = 50;

//	private static final String BINARY_FILE = "datasets/filmtrust.cf4j";
//
//	private static double REGULARIZATION = 0.095;
//	private static double LEARNING_RATE = 0.0035;
//	private static int GENS = 150;
//	private static double PBMUT = 0.1;
//	private static double PBX = 0.5;
//	private static int POP_SIZE = 50;


	private static int NUM_ITERS = 100;

	private static PrintWriter output;
	private static DataModel model;

	// Tree rewriting system
    private final static TreeRewriter<Op<Double>> trs = TreeRewriter.concat(
            compile("+($x,Zero) -> $x"),
            compile("+(Zero,$x) -> $x"),
            compile("-($x,Zero) -> $x"),
            compile("-(Zero,$x) -> --($x)"),
            compile("*(Zero,$x) -> Zero"),
            compile("*($x,Zero) -> Zero"),
            compile("*($x,One) -> $x"),
            compile("*(One,$x) -> $x"),
            compile("--(Zero) -> Zero"),
            compile("pow($x,Zero) -> One"),
            compile("pow($x,One) -> $x"),
            compile("pow(Zero,$x) -> Zero"),
            compile("pow(One,$x) -> One"),
            compile("inv(One) -> One"),
            compile("log(One) -> Zero"),
            compile("exp(Zero) -> One"),
            compile("log(exp($x)) -> $x")
    );

    private static TreeRewriter<Op<Double>> compile(final String rule) {
        return TreeRewriteRule.parse(rule, CustomMathOp::toMathOp);
    }

    // Multi-objective optimization
    final static VecFactory<double[]> fitfactory = VecFactory.ofDoubleVec(
            Optimize.MINIMUM,   // MAE
            Optimize.MAXIMUM,   // Novelty
            Optimize.MINIMUM    // Diversity
    );

	public static void main (String [] args) {
		CommandLineParser parser = new DefaultParser();
		Options options = new Options();
		options.addOption(new Option( "help", "print this message" ));
		options.addOption(
				Option.builder("lambda")
				.longOpt("lambda")
				.desc(String.format("default: %.6f", REGULARIZATION))
				.hasArg()
				.argName("VALUE")
				.type(double.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("gamma")
				.longOpt("gamma")
				.desc(String.format("default: %.6f", LEARNING_RATE))
				.hasArg()
				.argName("VALUE")
				.type(double.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("iters")
				.longOpt("iters")
				.desc("default: " + NUM_ITERS)
				.hasArg()
				.argName("VALUE")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("generations")
				.longOpt("generations")
				.desc("Number of generations, default: " + GENS)
				.hasArg()
				.argName("N")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("K")
				.desc("Number of topics, default: " + NUM_TOPICS)
				.hasArg()
				.argName("K")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("pop")
				.desc("Population size, default: " + POP_SIZE)
				.hasArg()
				.longOpt("population-size")
				.argName("pop")
				.type(int.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("pbx")
				.desc("Crossover probability, default: " + PBX)
				.hasArg()
				.longOpt("crossover-prob")
				.argName("pbx")
				.type(double.class)
				.valueSeparator()
				.build());
		options.addOption(
				Option.builder("pbmut")
				.desc("Mutation probability, default: " + PBMUT)
				.hasArg()
				.longOpt("mutation-prob")
				.argName("pbmut")
				.type(double.class)
				.valueSeparator()
				.build());

		try {
			CommandLine line = parser.parse(options, args);
			if (line.hasOption("help")){
				HelpFormatter formatter = new HelpFormatter();
				formatter.printHelp("gmf", options);
				return;
			}

			if (line.hasOption("lambda")){
				REGULARIZATION = Double.parseDouble(line.getOptionValue("lambda"));
			}
			if (line.hasOption("gamma")){
				LEARNING_RATE = Double.parseDouble(line.getOptionValue("gamma"));
			}
			if (line.hasOption("iters")){
				NUM_ITERS = Integer.parseInt(line.getOptionValue("iters"));
			}
			if (line.hasOption("generations")){
				GENS = Integer.parseInt(line.getOptionValue("generations"));
			}
			if (line.hasOption("K")) {
				NUM_TOPICS = Integer.parseInt(line.getOptionValue("K"));
			}
			if (line.hasOption("pop")) {
				POP_SIZE = Integer.parseInt(line.getOptionValue("pop"));
			}
			if (line.hasOption("pbx")) {
				PBX = Double.parseDouble(line.getOptionValue("pbx"));
			}
			if (line.hasOption("pbmut")) {
				PBMUT = Double.parseDouble(line.getOptionValue("pbmut"));
			}
		} catch (ParseException e) {
			System.out.println( "Unexpected exception:" + e.getMessage() );
		}

        DataSet ml100k = null;
        try {
            ml100k = new RandomSplitDataSet(BINARY_FILE, 0.2f, 0.2f, "::");
        } catch (IOException e) {
            System.out.println("There was an error when loading file " + BINARY_FILE);
            e.printStackTrace();
            System.exit(-1);
        }
        model = new DataModel(ml100k);

        final ISeq<Op<Double>> operations = ISeq.of(
                CustomMathOp.SIN,
                CustomMathOp.COS,
                CustomMathOp.ATAN,
                CustomMathOp.EXP,
                CustomMathOp.LOG,
                CustomMathOp.INV,
                CustomMathOp.NEG,
                CustomMathOp.ADD,
                CustomMathOp.SUB,
                CustomMathOp.MUL,
                CustomMathOp.POW
        );

		ISeq<Op<Double>> inputs = ISeq.empty();
		// Terminal nodes: variables
		for (int i = 0; i < NUM_TOPICS; i++) {
			inputs = inputs.append(
					Var.of("pu"+i, i),
					Var.of("qi"+i, i + NUM_TOPICS)
			);
		}
		inputs = inputs.append(CustomMathOp.Zero, CustomMathOp.One);

		// Tree building
		final Codec<ProgramGene<Double>, ProgramGene<Double>> codec = Codec.of(
				Genotype.of(ProgramChromosome.of(
						6,
						ch -> ch.root().size() <= 150,
						operations,
						inputs
				)), Genotype::gene
		);

		final Engine<ProgramGene<Double>, Vec<double[]>> engine = Engine
				.builder(GeneticProgramingOptimization::fitness, codec)
				.minimizing()
				.offspringSelector(new TournamentSelector<>())
				.alterers(
						new SingleNodeCrossover<>(PBX),
						new Mutator<>(PBMUT),
                        new MathRewriteAlterer<>(trs,1))
				.survivorsSelector(NSGA2Selector.ofVec())
				.populationSize(POP_SIZE)
                .executor(Runnable::run)
				.build();

		// Output file with unique filename
		SimpleDateFormat df = new SimpleDateFormat("yyyyMMddhhmmssSSS");
		try {
			Date d = new Date();
			File outputFile = new File(df.format(d) + ".csv");
			output = new PrintWriter(outputFile);
			output.println("Generation;MAE;Novelty;Diversity;Best");
		} catch (IOException e) {
			e.printStackTrace();
		}

		final ISeq<Phenotype<ProgramGene<Double>,Vec<double[]>>> bestFromPareto =
                RandomRegistry.with(new Random(SEED), r ->
                    engine.stream()
                    .limit(Limits.byFixedGeneration(GENS))
                    .peek(GeneticProgramingOptimization::update)
                    .peek(GeneticProgramingOptimization::toFile)
					.collect(MOEA.toParetoSet(NUM_RESULTS))
                );

		output.close();

		System.out.println("\n------------ Best " + NUM_RESULTS + " individuals from Pareto Set ------------");

		bestFromPareto.stream().forEach(individual -> System.out.println(
					individual
					.genotype()
					.gene()
					.toParenthesesString()
					.replace("(", " ")
					.replace(")", " ")
					.replace(",", " ")
				)
			);
	}

	private static Vec<double[]> fitness(final ProgramGene<Double> program) {
		String func = program.toParenthesesString()
				.replace("("," ")
				.replace(")"," ")
				.replace(",", " ");

		Recommender emf = new EMF(func, model, NUM_TOPICS, NUM_ITERS, REGULARIZATION,
                LEARNING_RATE, SEED,false);
		emf.fit();
		QualityMeasure mae = new MAE(emf);
		QualityMeasure novelty = new Novelty(emf, NUM_RECS);
		QualityMeasure diversity = new Diversity(emf, NUM_RECS);
		double error = mae.getScore();
		double nov = novelty.getScore();
		double div = diversity.getScore();

		return fitfactory.newVec(new double[]{
		            (Double.isNaN(error) || Double.isInfinite(error)) ? 10.0 : error,
                    (Double.isNaN(nov) || Double.isInfinite(nov)) ? 0.0 : nov,
                    (Double.isNaN(div) || Double.isInfinite(div)) ? 0.0 : div,
                }
        );
	}

	private static void update(final EvolutionResult<ProgramGene<Double>, Vec<double[]>> result) {
		List<EvolutionResult<ProgramGene<Double>, Vec<double[]>>> l = new ArrayList<>();
		l.add(result);
		ISeq<Phenotype<ProgramGene<Double>, Vec<double[]>>> paretoSet = l.stream().collect(MOEA.toParetoSet(NUM_RESULTS));

		String info = String.format(
				"%d/%d:\tN-pareto=%4d\tHypervolume=%6.4f | %6.4f\tinvalids=%3d",
                result.generation(),
				GENS,
                paretoSet.length(),
//				MultiobjectiveStatistics.hypervolume(paretoSet, Vec.of(20.0, 0.0, 1.0)),
				result.bestFitness().data()[2],
                result.invalidCount());
		System.out.println(info);
	}

	private static void toFile(final EvolutionResult<ProgramGene<Double>, Vec<double[]>> result) {
		result.population().stream().forEach(individual ->
                output.println(result.generation() + ";" +
                        individual.fitness().data()[0] + ";" +
						individual.fitness().data()[1] + ";" +
						individual.fitness().data()[2] + ";" +
                        individual.genotype().gene().toParenthesesString())
                );
	}
}

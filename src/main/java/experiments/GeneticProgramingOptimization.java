package main.java.experiments;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.DataSet;
import es.upm.etsisi.cf4j.data.RandomSplitDataSet;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.qualityMeasure.prediction.MAE;
import es.upm.etsisi.cf4j.recommender.Recommender;
import io.jenetics.*;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.ext.SingleNodeCrossover;
import io.jenetics.ext.rewriting.TreeRewriteRule;
import io.jenetics.ext.rewriting.TreeRewriter;
import io.jenetics.prog.MathRewriteAlterer;
import io.jenetics.prog.ProgramChromosome;
import io.jenetics.prog.ProgramGene;
import io.jenetics.prog.op.Op;
import io.jenetics.prog.op.Var;
import io.jenetics.util.ISeq;
import main.java.mf.EMF;
import org.apache.commons.cli.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.stream.Collectors;

public class GeneticProgramingOptimization {

	private static final String BINARY_FILE = "datasets/ml100k.dat";
	private static final long SEED = 1337;

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

        DataSet ml100k = new RandomSplitDataSet(BINARY_FILE, 0.2f, 0.2f, "::");
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

		final Engine<ProgramGene<Double>, Double> engine = Engine
				.builder(GeneticProgramingOptimization::fitness, codec)
				.minimizing()
				.offspringSelector(new TournamentSelector<>())
				.alterers(
						new SingleNodeCrossover<>(PBX),
						new Mutator<>(PBMUT),
                        new MathRewriteAlterer<>(trs,1))
				.survivorsSelector(new EliteSelector<ProgramGene<Double>, Double>(2, new MonteCarloSelector<>()))
				.populationSize(POP_SIZE)
                .executor((Executor)Runnable::run)
				.build();

		// Output file with unique filename
		SimpleDateFormat df = new SimpleDateFormat("ddMMyy-hhmmss.SSS");
		try {
			Date d = new Date();
			File outputFile = new File(df.format(d) + ".csv");
			output = new PrintWriter(outputFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

		final EvolutionResult<ProgramGene<Double>, Double> population = engine.stream()
				.limit(GENS)
				.peek(GeneticProgramingOptimization::update)
				.peek(GeneticProgramingOptimization::toFile)
				.collect(EvolutionResult.toBestEvolutionResult());

//		try {
//			if (popOutput.canWrite())
//				IO.object.write(population.getPopulation(), popOutput);
//			output.flush();
//			output.close();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

		output.close();

		System.out.println(population.bestPhenotype().genotype().gene().toParenthesesString());
	}

	private static double fitness(final ProgramGene<Double> program) {
		String func = program.toParenthesesString()
				.replace("("," ")
				.replace(")"," ")
				.replace(",", " ");

		Recommender emf = new EMF(func, model, NUM_TOPICS, NUM_ITERS, REGULARIZATION, LEARNING_RATE, SEED,false);
		QualityMeasure mae = new MAE(emf);
		double error = mae.getScore(1);

		return (Double.isNaN(error) || Double.isInfinite(error)) ? 10.0 : error;
	}

	private static void update(final EvolutionResult<ProgramGene<Double>, Double> result) {
		String info = String.format(
				"%d/%d:\tbest=%.4f\tinvalids=%d\tavg=%.4f\tbest=%s",
                result.generation(),
				GENS,
                result.bestFitness(),
                result.invalidCount(),
				result.population().stream().collect(Collectors.averagingDouble(Phenotype::fitness)),
				result.bestPhenotype().genotype().gene().toParenthesesString());
		System.out.println(info);
	}

	private static void toFile(final EvolutionResult<ProgramGene<Double>, Double> result) {
		double[] fitnesses = result.population().stream().mapToDouble(Phenotype::fitness).toArray();
		List<CharSequence> fs = new ArrayList<>();

		for (double fitness : fitnesses) {
			fs.add(String.valueOf(fitness));
		}

		output.println(result.generation() + "," + String.join(",", fs));

	}
}

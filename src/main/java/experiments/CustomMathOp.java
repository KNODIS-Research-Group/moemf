package main.java.experiments;

import io.jenetics.prog.op.Const;
import io.jenetics.prog.op.Op;
import io.jenetics.prog.op.Var;

import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.lang.Math.*;
import static java.util.Objects.requireNonNull;

public enum CustomMathOp implements Op<Double> {
    NEG("--", 1, v -> -v[0]),
    ADD("+", 2, v -> v[0] + v[1]),
    SUB("-", 2, v -> v[0] - v[1]),
    MUL("*", 2, v -> v[0]*v[1]),
    INV("inv", 1, v -> 1.0/v[1]),
    POW("pow", 2, v -> pow(v[0], v[1])),
    EXP("exp", 1, v -> exp(v[0])),
    LOG("log", 1, v -> log(v[0])),
    SIN("sin", 1, v -> sin(v[0])),
    COS("cos", 1, v -> cos(v[0])),
    ATAN("atan", 1, v -> atan(v[0]));

    /* *************************************************************************
     * Mathematical constants.
     * ************************************************************************/
    public static final Const<Double> One = Const.of("One", 1.0);
    public static final Const<Double> Zero = Const.of("Zero", 0.0);


    private final String _name;
    private final int _arity;
    private final Function<Double[], Double> _function;

    private CustomMathOp(
            final String name,
            final int arity,
            final Function<Double[], Double> function
    ) {
        assert name != null;
        assert arity >= 0;
        assert function != null;

        _name = name;
        _function = function;
        _arity = arity;
    }

    @Override
    public int arity() {
        return _arity;
    }

    @Override
    public Double apply(final Double[] args) {
        return _function.apply(args);
    }

    public double eval(final double... args) {
        return apply(
                DoubleStream.of(args)
                        .boxed()
                        .toArray(Double[]::new)
        );
    }

    @Override
    public String toString() {
        return _name;
    }

    public static Op<Double> toMathOp(final String string) {
        requireNonNull(string);

        final Op<Double> result;
        final Optional<Const<Double>> cop = toConst(string);
        if (cop.isPresent()) {
            result = cop.orElseThrow(AssertionError::new);
        } else {
            final Optional<Op<Double>> mop = toOp(string);
            result = mop.isPresent()
                    ? mop.orElseThrow(AssertionError::new)
                    : Var.parse(string);
        }

        return result;
    }

    static Optional<Const<Double>> toConst(final String string) {
        //return Numbers.toDoubleOptional(string)
                //.map(Const::of);
        if (string.equals("One")) {
            return Optional.of(One);
        } else if (string.equals("Zero")) {
            return Optional.of(Zero);
        } else {
            return Optional.empty();
        }
    }

    private static Optional<Op<Double>> toOp(final String string) {
        return Stream.of(values())
                .filter(op -> Objects.equals(op._name, string))
                .map(op -> (Op<Double>)op)
                .findFirst();
    }
}

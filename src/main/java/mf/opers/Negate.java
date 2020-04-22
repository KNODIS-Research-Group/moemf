package mf.opers;

import org.moeaframework.util.tree.Environment;
import org.moeaframework.util.tree.Node;
import org.moeaframework.util.tree.NumberArithmetic;

public class Negate extends Node {
    public Negate() {
        super(Number.class, Number.class);
    }

    public Node copyNode() {
        return new Negate();
    }

    public Object evaluate(Environment environment) {
        return NumberArithmetic.mul(
                (Number)getArgument(0).evaluate(environment),
                -1.0
        );
    }
}

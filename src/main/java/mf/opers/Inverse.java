package mf.opers;

import org.moeaframework.util.tree.Environment;
import org.moeaframework.util.tree.Node;
import org.moeaframework.util.tree.NumberArithmetic;

public class Inverse extends Node {
    public Inverse() {
        super(Number.class, Number.class);
    }

    public Node copyNode() {
        return new Inverse();
    }

    public Object evaluate(Environment environment) {
        return NumberArithmetic.div(
                1.0,
                (Number)getArgument(0).evaluate(environment)
        );
    }
}

package Part2;

import org.jgap.gp.GPFitnessFunction;
import org.jgap.gp.IGPProgram;
import org.jgap.gp.terminal.Variable;

public class FitnessFunction extends GPFitnessFunction {

    Object[] NO_ARGS = new Object[0];
    Variable _xVariable;

    double[] input;
    double[] output;





    public FitnessFunction(double[] input, double[] output, Variable x) {
        this.input=input;
        this.output=output;
        this._xVariable=x;

    }

    @Override
    protected double evaluate(IGPProgram igpProgram) {
        double result = 0.0;
        double longResult = 0.0;
        for (int i = 0; i < input.length; i++) {
            // Set the input values
            _xVariable.set(input[i]);
            // Execute the genetically engineered algorithm
            double value = igpProgram.execute_double(0, NO_ARGS);

            // The closer longResult gets to 0 the better the algorithm.
            longResult += Math.abs(value - output[i]);
            if (Double.isInfinite(longResult)) {
                return Double.MAX_VALUE;
            }
        }
        if (longResult < 0.001) {
            longResult = 0.0;
        }
        result = longResult;
        return result;
    }
}

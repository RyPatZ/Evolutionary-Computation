package Part1;

import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    public  double[] hidden_layer_bias;
    public  double[] output_layer_bias;

    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;


    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.hidden_layer_bias= new double[]{-0.02, -0.20};
        this.output_layer_bias= new double[]{-0.33, 0.26,0.06};
        this.learning_rate = learning_rate;
    }


    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = 1/(1+Math.exp(-input)); //TODO!
        return output;
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;

            int count =0;
            for (double input : inputs){
                double weight = hidden_layer_weights[count][i];
                weighted_sum += weight* input;
                count++;
            }
            weighted_sum += hidden_layer_bias[i];
            output= sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output.
            double weighted_sum = 0;
            double output = 0;
            int count =0;
            for (double input : hidden_layer_outputs){
                double weight = output_layer_weights[count][i];
                weighted_sum += weight* input;
                count++;
            }
            weighted_sum += output_layer_bias[i];
            output= sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }

        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs) {

        double[] output_layer_betas = new double[num_outputs];
        // TODO! Calculate output layer betas.

        int[] desired_op=new int[3];
        if(desired_outputs==0){
            desired_op[0]=1;
            desired_op[1]=0;
            desired_op[2]=0;
        }
        if(desired_outputs==1){
            desired_op[0]=0;
            desired_op[1]=1;
            desired_op[2]=0;
        }
        if(desired_outputs==2){
            desired_op[0]=0;
            desired_op[1]=0;
            desired_op[2]=1;
        }

        int count =0;
        for(double output_l_output: output_layer_outputs){
            double beta = desired_op[count] - output_l_output;
            output_layer_betas [count] =beta;
            count++;
        }
        System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];
        // TODO! Calculate hidden layer betas.
        count =0;
        for(double hidden_l_output: hidden_layer_outputs){
            double beta =0;
            for(int i =0; i<output_layer_outputs.length;i++){
                double weight =output_layer_weights[count][i];
                beta += weight * output_layer_outputs[i]*(1-output_layer_outputs[i])*output_layer_betas[i];
            }
            hidden_layer_betas[count]=beta;
            count ++;
        }

       System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        // TODO! Calculate output layer weight changes.

        double learningRate = 0.2;
        count =0;
        for(double hidden_l_output: hidden_layer_outputs){
            for(int i =0; i<output_layer_outputs.length;i++){
                double weight =output_layer_weights[count][i];
                double weightChange=learningRate*hidden_l_output*output_layer_outputs[i]*(1-output_layer_outputs[i])*output_layer_betas[i];
                delta_output_layer_weights[count][i] = weightChange;
            }
            count++;
        }


        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        // TODO! Calculate hidden layer weight changes.
        count=0;
        for(double input: inputs){
            for(int i =0; i<hidden_layer_outputs.length;i++){
                double weightChange=learningRate*input*hidden_layer_outputs[i]*(1-hidden_layer_outputs[i])*hidden_layer_betas[i];
                delta_hidden_layer_weights[count][i] = weightChange;
            }
            count++;
        }

        double[] delta_hidden_layer_bias = new double[num_hidden];
        count=0;
        for(double hidden_l_output: hidden_layer_outputs){
            double bia = learningRate*hidden_l_output*(1-hidden_l_output)*hidden_layer_bias[count];
            delta_hidden_layer_bias[count]=bia;
            count++;
        }

        double[] delta_output_layer_bias = new double[num_outputs];
        count=0;
        for(double output_l_output: hidden_layer_outputs){
            double bia = learningRate*output_l_output*(1-output_l_output)*output_layer_bias[count];
            delta_hidden_layer_bias[count]=bia;
            count++;
        }

        count = 0;
        for(double bia : delta_hidden_layer_bias){
            hidden_layer_bias[count]=bia;
        }
        count = 0;
        for(double bia : delta_output_layer_bias){
            output_layer_bias[count]=bia;
        }


        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        // TODO! Update the weights
        int count =0;
        for(double [] olWeights : output_layer_weights ){
            int count1=0;
            for(double olWeight: olWeights){
                output_layer_weights[count][count1] += delta_output_layer_weights[count][count1];
                count1++;
            }
            count++;
        }

        count =0;
        for(double [] hlweights : hidden_layer_weights ){
            int count1=0;
            for(double hlWeight: hlweights){
                hidden_layer_weights[count][count1] += delta_hidden_layer_weights[count][count1];
                count1++;
            }
            count++;
        }

        System.out.println("Placeholder");
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            int [] predict= predict(instances);
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
                int predicted_class = predict[i]; // TODO!
                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // TODO: Print accuracy achieved over this epoch
            double acc = Double.NaN;
            int count = 0;
            int predictionCount=0;
            for (int de:desired_outputs){
                if(de==predictions[predictionCount]){
                    count++;
                }
                predictionCount++;
            }
            acc= (double) count/instances.length*100;
            System.out.println("-------------------------------------------------------------------------------------------");
            System.out.println("After "+epochs +" of training\n We got "+ count + " out of " + instances.length +" correct");
            System.out.println("Accuracy = " + acc +"%");
            System.out.println("-------------------------------------------------------------------------------------------");
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            double[] op = outputs[1];
                if(op[0]>op[1] && op[0]>op[2]){
                    predicted_class=0;
                }
                else if(op[1]>op[0] && op[1]>op[2]){
                    predicted_class=1;
                }
                else{
                    predicted_class=2;
                }

            predictions[i] = predicted_class;
        }
        return predictions;
    }

}

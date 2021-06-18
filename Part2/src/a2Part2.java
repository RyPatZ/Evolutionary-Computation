package Part2;

import Part1.Util;
import org.jgap.*;
import org.jgap.gp.*;
import org.jgap.gp.impl.*;
import org.jgap.gp.function.*;
import org.jgap.gp.terminal.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Scanner;


public class a2Part2 extends GPProblem {
    double[] x;
    double[] y;
    Variable _xVariable;

    public a2Part2() throws InvalidConfigurationException {

        super(new GPConfiguration());
        x = new double[20];
        y = new double[20];
        readFromFile();

        GPConfiguration config = getGPConfiguration();
        _xVariable = Variable.create(config, "X", CommandGene.DoubleClass);
        config.setGPFitnessEvaluator(new DeltaGPFitnessEvaluator());
        config.setMaxInitDepth(4);
        config.setPopulationSize(1000);
        config.setMaxCrossoverDepth(8);
        config.setFitnessFunction(new FitnessFunction(x, y,  _xVariable));
        config.setStrictProgramCreation(true);
    }

    @Override
    public GPGenotype create() throws InvalidConfigurationException {
        GPConfiguration config = getGPConfiguration();

        // The return type of the GP program.
        Class[] types = { CommandGene.DoubleClass };

        // Arguments of result-producing chromosome: none
        Class[][] argTypes = { {} };

        // Next, we define the set of available GP commands and terminals to
        // use.
        CommandGene[][] nodeSets = {
                {
                        _xVariable,
                        new Add(config, CommandGene.DoubleClass),
                        new Subtract(config, CommandGene.DoubleClass),
                        new Multiply(config, CommandGene.DoubleClass),
                        new Divide(config, CommandGene.DoubleClass),
                        new Terminal(config, CommandGene.DoubleClass, 0.0, 10.0, true)
                }
        };

        GPGenotype result = GPGenotype.randomInitialGenotype(config, types, argTypes,
                nodeSets, 20, true);

        return result;
    }

    public void readFromFile(){
        try {
            String file = "regression.txt";
            FileReader fileReader = new FileReader(new File(file));
            BufferedReader br = new BufferedReader(fileReader);
            br.readLine();
            br.readLine();
            String line;
            int count =0;
            while ((line = br.readLine()) != null) {
                String[] split = line.split("\\s+");
                x[count]=Double.parseDouble(split[1]);
                y[count]=Double.parseDouble(split[2]);
                count++;
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws Exception {
        GPProblem gpproblem = new a2Part2();
        GPGenotype gp = gpproblem.create();
        gp.setVerboseOutput(true);
        //stop criteria
        //if the fitness value doesn't improve for 25 evolution and the fitness value =0 then stop
        double smallest_fitnessvalue=gp.getFittestProgram().getFitnessValue();
        int count =0;
        for(int i=0; i<= 200;i++) {
            gp.evolve(1);
            double fitness= gp.getFittestProgramComputed().getFitnessValue();
            if(fitness<smallest_fitnessvalue){
                smallest_fitnessvalue=fitness;
            }
            else {
                count++;
            }
            if(count==100||(gp.getFittestProgramComputed().getFitnessValue()==0)){
                break;
            }
        }

        System.out.println("Formula to discover:");
        gp.outputSolution(gp.getAllTimeBest());
    }
}

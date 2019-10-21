package org.deeplearning4j.examples.dataexamples;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Java_model {
    private static Logger log = LoggerFactory.getLogger(ImportKerasConfig.class);

    public static void main(String[] args) throws Exception {

        KerasLayer.registerLambdaLayer("lambda_4", new SameDiffLambdaLayer()
        {
            @Override
            public SDVariable defineLayer(SameDiff sameDiff, SDVariable sdVariable)
            {
                return sameDiff.squeeze(sdVariable, -1);
            }

            @Override
            public InputType getOutputType(int layerIndex, InputType inputType)
            {
                return InputType.feedForward(15);
            }
        });

        KerasLayer.registerLambdaLayer("lambda_3", new SameDiffLambdaLayer()
        {
            @Override
            public SDVariable defineLayer(SameDiff sameDiff, SDVariable sdVariable)
            {
                return sameDiff.stridedSlice(sdVariable, new int[]{ 0, 0, 10 }, new int[]{ (int)sdVariable.getShape()[0], (int)sdVariable.getShape()[1], (int)sdVariable.getShape()[2]-10}, new int[]{ 1, 1, 1 });
            }

            @Override
            public InputType getOutputType(int layerIndex, InputType inputType)
            {
                return InputType.recurrent(100);
            }
        });

        KerasLayer.registerLambdaLayer("lambda_2", new SameDiffLambdaLayer()
        {
            @Override
            public SDVariable defineLayer(SameDiff sameDiff, SDVariable sdVariable)
            {
                return sameDiff.stridedSlice(sdVariable, new int[]{ 0, 0, 10 }, new int[]{ (int)sdVariable.getShape()[0], (int)sdVariable.getShape()[1], (int)sdVariable.getShape()[2]-10}, new int[]{ 1, 1, 1 });
            }

            @Override
            public InputType getOutputType(int layerIndex, InputType inputType)
            {
                return InputType.recurrent(60);
            }
        });

        KerasLayer.registerLambdaLayer("lambda_1", new SameDiffLambdaLayer()
        {
            @Override
            public SDVariable defineLayer(SameDiff sameDiff, SDVariable sdVariable)
            {
                return sameDiff.nn().pad(sdVariable, new int[][]{ { 0, 0 }, { 10, 10 }}, 1);
            }

            @Override
            public InputType getOutputType(int layerIndex, InputType inputType)
            {
                return InputType.feedForward(20);
            }
        });

        ComputationGraph model = org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasModelAndWeights
            ("/home/model1_functional.h5");

        INDArray myArray = Nd4j.zeros(1,4); // one row 4 column array
        myArray.putScalar(0,0,1);
        myArray.putScalar(0,1,3);
        myArray.putScalar(0,2,4);
        myArray.putScalar(0,3,8);

        INDArray output = model.outputSingle(myArray);
        System.out.println("First model output");
        System.out.println(output);

    }

}

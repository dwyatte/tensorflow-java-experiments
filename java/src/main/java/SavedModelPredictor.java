import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

public class SavedModelPredictor {

    public static final String savedModelPath = "/Users/dwyatte/GitHub/tensorflow-java-experiments/python/export/multimodal/0";
    public static final String tagSet = "serve";
    public static final String floatInputName = "serving_default_float_input";
    public static final String stringInputName = "serving_default_string_input";
    public static final String outputName = "StatefulPartitionedCall_1";

    public static void main(String[] args) throws Exception {
        SavedModelBundle savedModel = SavedModelBundle.load(savedModelPath, tagSet);
        Tensor floatInputTensor = Tensor.create(new float[][] {
            {0.0f}, {1.0f}
        });
        Tensor stringInputTensor = Tensor.create(new byte[][] {
            "a sentence".getBytes(),
            "b sentence".getBytes(),
        });
        Tensor outputTensor = savedModel.session().runner()
                                .feed(floatInputName, floatInputTensor)
                                .feed(stringInputName, stringInputTensor)
                                .fetch(outputName).run().get(0);
        float[][] output = new float[2][1];
        outputTensor.copyTo(output);
        System.out.println(output[0][0]);
    }
}

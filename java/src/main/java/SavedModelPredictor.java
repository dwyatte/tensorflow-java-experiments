import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

public class SavedModelPredictor {

    public static final String savedModelPath = "/Users/dwyatte/GitHub/tensorflow-java-experiments/python/export/use/0";
    public static final String tagSet = "serve";
    public static final String inputName = "serving_default_input_1";
    public static final String outputName = "StatefulPartitionedCall_1";

    public static void main(String[] args) throws Exception {
        SavedModelBundle savedModel = SavedModelBundle.load(savedModelPath, tagSet);
        // Tensor inputTensor = Tensor.create(0.0f);
        Tensor inputTensor = Tensor.create(new byte[][] {
            "a sentence".getBytes(),
            "b sentence".getBytes(),
        });
        Tensor outputTensor = savedModel.session().runner().feed(inputName, inputTensor).fetch(outputName).run().get(0);
        float[][] output = new float[2][1];
        outputTensor.copyTo(output);
        System.out.println(output[0][0]);
    }
}

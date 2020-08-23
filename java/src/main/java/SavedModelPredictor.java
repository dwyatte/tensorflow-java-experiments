import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

public class SavedModelPredictor {

    public static final String savedModelPath = "/Users/dwyatte/GitHub/tensorflow-java-experiments/python/models/export/tf2/0";
    public static final String tagSet = "serve";
    public static final String inputName = "serving_default_input_1";
    public static final String outputName = "PartitionedCall";

    public static void main(String[] args) throws Exception {
        SavedModelBundle savedModel = SavedModelBundle.load(savedModelPath, tagSet);
        Tensor inputTensor = Tensor.create(0.0f);
        Tensor outputTensor = savedModel.session().runner().feed(inputName, inputTensor).fetch(outputName).run().get(0);
        System.out.println(outputTensor.floatValue());
    }
}

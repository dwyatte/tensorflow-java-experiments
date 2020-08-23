import org.tensorflow.SavedModelBundle;

public class SavedModelPredictor {

    public static final String savedModelPath = "/Users/dwyatte/GitHub/tensorflow-java-experiments/python/models/export/tf2/0";
    public static final String tagSet = "serve";

    public static void main(String[] args) throws Exception {
        SavedModelBundle savedModel = SavedModelBundle.load(savedModelPath, tagSet);

    }
}

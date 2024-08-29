package weka.classifiers.trees;
import weka.classifiers.RandomizableClassifier;
import weka.core.*;
import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

public class XGBoostTree extends RandomizableClassifier implements WeightedInstancesHandler {

    // Interface for representing nodes in the tree
    /** A possible way to represent the tree structure using Java records. */
    private interface Node {
    }
    private record InternalNode(Attribute attribute, double splitPoint, Node leftSuccessor, Node rightSuccessor)
            implements Node, Serializable { }
    private record LeafNode(double prediction) implements Node, Serializable { }

    /** The root node of the decision tree. */
    private Node rootNode = null;
    /** The training instances. */
    private Instances data;
    /** A class for objects that hold a split specification, including the quality of the split. */
    private class SplitSpecification {
        private final Attribute attribute; private double splitPoint; private double splitQuality;
        private SplitSpecification(Attribute attribute, double splitQuality, double splitPoint) {
            this.attribute = attribute; this.splitQuality = splitQuality; this.splitPoint = splitPoint;
        }
    }

    /**
     * A class for objects that contain the sufficient statistics required to measure split quality,
     * These statistics are sufficient to compute the sum of gradients, Hessians and other information needed.
     */
    // Sufficient statistics or summary statistics to store the sum of gradients, Hessians and other information needed for split calc for purity

    private class SufficientStatistics {
        private int n = 0; private double sumOfGradient = 0.0; private double sumOfHessian = 0.0;
        private SufficientStatistics(int n, double sumOfGradient, double sumOfHessian) {
            this.n = n; this.sumOfGradient = sumOfGradient; this.sumOfHessian = sumOfHessian;
        }
        private void updateStats(double input_gradient, double input_hessian, boolean add) {
            n = (add) ? n + 1 : n - 1;
            sumOfGradient = (add) ? sumOfGradient + input_gradient : sumOfGradient - input_gradient;
            sumOfHessian = (add) ? sumOfHessian + input_hessian : sumOfHessian - input_hessian;
        }
    }

    /** Computes the info gain for branch (left, right or before) based on the sufficient statistics */
    private double computeBranchInfoGain(SufficientStatistics stats)
    {
        return ((stats.sumOfGradient * stats.sumOfGradient) / (stats.sumOfHessian + lambda));
    }

    // Method to calculate split quality based on the provided equation refer that ppt in lecture and also in research paper 2016
    private double splitQuality(SufficientStatistics initialSufficientStatistics,
                                SufficientStatistics statsLeft, SufficientStatistics statsRight)
    {
        // Ensure splits are only considered if the child Hessians <= minimum weight requirement
        if (min_child_weight >= statsLeft.sumOfHessian || min_child_weight >= statsRight.sumOfHessian)
        {
            return Double.NEGATIVE_INFINITY;
        }
        // calculation of split quality based on the formula given double gain =

        return (0.5 * (
                computeBranchInfoGain(statsLeft) +
                        computeBranchInfoGain(statsRight) -
                        computeBranchInfoGain(initialSufficientStatistics)
        ) - gamma);

    }
    /**
     * Finds the best split point and returns the corresponding split specification object. The given indices
     * define the subset of the training set for which the split is to be found. The initialStats are the sufficient
     * statistics before the data is split.
     */

    // Method to find the best split point for an attribute, with column subsampling
    private SplitSpecification findBestSplitPoint(int[] indices, Attribute attribute,
                                                  SufficientStatistics initialStats) {
        var statsLeft = new SufficientStatistics(0, 0.0, 0.0);
        var statsRight = new SufficientStatistics(initialStats.n, initialStats.sumOfGradient, initialStats.sumOfHessian);
        var splitSpecification = new SplitSpecification(attribute, 0.0, Double.NEGATIVE_INFINITY);
        var previousValue = Double.NEGATIVE_INFINITY;

        // Sorting the indices based on the attribute values which sort order


        for (int i : Arrays.stream(Utils.sortWithNoMissingValues(
                        Arrays.stream(indices).mapToDouble(x -> data.instance(x).value(attribute)).toArray()))
                .map(x -> indices[x])
                .toArray()) {

            Instance instance = data.instance(i);
            if (instance.value(attribute) > previousValue) {
                var splitQuality = splitQuality(initialStats, statsLeft, statsRight);
                if (splitQuality > splitSpecification.splitQuality) {
                    splitSpecification.splitQuality = splitQuality;
                    splitSpecification.splitPoint = (instance.value(attribute) + previousValue) / 2.0;
                }
                previousValue = instance.value(attribute);
            }

            // Negation of gradient is done as we receive -g from XGBoost class for making calculation simple
            statsLeft.updateStats(-instance.classValue(), instance.weight(), true);
            statsRight.updateStats(-instance.classValue(), instance.weight(), false);
        }
        return splitSpecification;
    }
    /**
     * Recursively grows a tree for a given set of data.
     * Recursive method to build the decision tree - top-down approach and depth parameter gets added as one of the stopping criteria
     */

    private Node makeTree(int[] indices, int depth) {
        var stats = new SufficientStatistics(0, 0.0, 0.0);
        for (int i : indices) {
            stats.updateStats(-data.instance(i).classValue(), data.instance(i).weight(), true); /*gradient(negation is done to make it positive for simple calculation) and hessian value*/
        }

        /* stopping criteria is depth of tree, minimum child weights and no further split improvement
         in terms of accuracy and If the Hessian sum is zero or negative (stats.hessianSum <= 0) indicating numerical issues
         and instance at leaf node as 1 considered which can be change
         */
        if (stats.n <= min_instances_in_leaf || depth >= max_depth || min_child_weight >= stats.sumOfHessian || stats.sumOfHessian <= 0) {
            // Calculate leaf node prediction using the formula: -(Gradient / (Hessian + lambda)) * learning_factor and then apply shrinkage(learning factor
            return new LeafNode((-stats.sumOfGradient / (stats.sumOfHessian + lambda)) * eta);
        }

        var bestSplitSpecification = new SplitSpecification(null, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);

        // Column subsampling logic is integrated in the below code -

        /* reference taken for shuffling from - https://www.geeksforgeeks.org/collections-shuffle-method-in-java-with-examples/*/


        List<Attribute> attributes = Collections.list(data.enumerateAttributes());
        //System.out.println(data.enumerateAttributes());
        Random random = new Random();
        Collections.shuffle(attributes,random); // Shuffling attributes for random column selection
        int numSelectedAttributes = (int) Math.round(attributes.size() * colsample_bynode);
        //System.out.println(numSelectedAttributes);
        attributes = attributes.subList(0, numSelectedAttributes);

        // reference taken for sublist - https://www.javatpoint.com/java-list-sublist-method

        for (Attribute attribute : attributes) {
            var splitSpecification = findBestSplitPoint(indices, attribute, stats);
            if (splitSpecification.splitQuality > bestSplitSpecification.splitQuality) {
                bestSplitSpecification = splitSpecification;
            }
        }

        if (bestSplitSpecification.splitQuality < 1E-6) {
            return new LeafNode((-stats.sumOfGradient / (stats.sumOfHessian + lambda)) * eta);
        } else {
            var leftSubset = new ArrayList<Integer>(indices.length);
            var rightSubset = new ArrayList<Integer>(indices.length);
            for (int i : indices) {
                if (data.instance(i).value(bestSplitSpecification.attribute) < bestSplitSpecification.splitPoint) {
                    leftSubset.add(i);
                } else {
                    rightSubset.add(i);
                }
            }
            return new InternalNode(bestSplitSpecification.attribute, bestSplitSpecification.splitPoint,
                    makeTree(leftSubset.stream().mapToInt(Integer::intValue).toArray(), depth + 1),
                    makeTree(rightSubset.stream().mapToInt(Integer::intValue).toArray(), depth + 1));
        }


    }

    // Get capabilities for numeric attributes and numeric class
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        return result;
    }

    // Method for subsampling for instances - row subsampling

    private Instances subsample(Instances inputData, double rowsubsample)
    {
        // Calculate how many rows to sample in data
        int rowSampleSize = (int) Math.round(inputData.numInstances() * rowsubsample);
        Instances newSampledData = new Instances(inputData, rowSampleSize);
        // Create an array of indices
        int[] indices_array = IntStream.range(0, inputData.numInstances()).toArray();
        /* Fisher-Yates shuffle on indices - reference taken - https://www.geeksforgeeks.org/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/*/
        Random random = new Random();
        for (int i = indices_array.length - 1; i > 0; i--)
        {
            int j = random.nextInt(i + 1);
            // Swapping  the indices of the array
            int temp = indices_array[i];
            indices_array[i] = indices_array[j];
            indices_array[j] = temp;
        }

        // Pick the first 'rowSampleSize' rows after shuffling indices
        for (int i = 0; i < rowSampleSize; i++)
        {
            newSampledData.add(inputData.instance(indices_array[i]));
        }
        return newSampledData;
    }

    // Main method to build the classifier
    public void buildClassifier(Instances trainingData) throws Exception {
        // First, use the capabilities to check whether the learning algorithm can handle the data.
        getCapabilities().testWithFail(trainingData);
        // Row sampling is done here for generalization - kind of bagging features
        this.data = subsample(new Instances(trainingData), subsample);
        // System.out.println(this.data.numInstances());
        //decision tree creation process get start from the root node after sampling
        rootNode = makeTree(IntStream.range(0, this.data.numInstances()).toArray(), 0);
    }

    // Recursive method to predict an instance's class by traversing the tree
    private double makePrediction(Node node, Instance instance) {
        if (node instanceof LeafNode) {
            return ((LeafNode) node).prediction;
        } else if (node instanceof InternalNode) {
            if (instance.value(((InternalNode) node).attribute) < ((InternalNode) node).splitPoint) {
                return makePrediction(((InternalNode) node).leftSuccessor, instance);
            } else {
                return makePrediction(((InternalNode) node).rightSuccessor, instance);
            }
        }
        return Utils.missingValue(); // This should never happen
    }

    // Classify an instance by traversing the tree
    /**
     * Provides a prediction for the current instance by calling the recursive makePrediction(Node, Instance) method.
     */
    public double classifyInstance(Instance instance) {
        return makePrediction(rootNode, instance);
    }

    /** Recursively produces the string representation of a branch in the tree. */
    private void branchToString(StringBuffer sb, boolean left, int level, InternalNode node) {
        sb.append("\n");
        for (int j = 0; j < level; j++) { sb.append("|   "); }
        sb.append(node.attribute.name() + (left ? " < " : " >= ") + Utils.doubleToString(node.splitPoint, getNumDecimalPlaces()));
        toString(sb, level + 1, left ? node.leftSuccessor : node.rightSuccessor);
    }

    /**
     * Recursively produces a string representation of a subtree by calling the branchToString(StringBuffer, int,
     * Node) method for both branches, unless we are at a leaf.
     */
    private void toString(StringBuffer sb, int level, Node node) {
        if (node instanceof LeafNode) {
            sb.append(": " + ((LeafNode) node).prediction);
        } else {
            branchToString(sb, true, level, (InternalNode) node);
            branchToString(sb, false, level, (InternalNode) node);
        }
    }

    /**
     * Returns a string representation of the tree by calling the recursive toString(StringBuffer, int, Node) method.
     */
    public String toString() {
        if (rootNode == null) {
            return "No model has been built yet.";
        }
        StringBuffer sb = new StringBuffer();
        toString(sb, 0, rootNode);
        return sb.toString();
    }

    /** The hyperparameters for an XGBoost tree. */
    private double eta = 0.3;
    @OptionMetadata(displayName = "eta", description = "eta",
            commandLineParamName = "eta", commandLineParamSynopsis = "-eta <double>", displayOrder = 1)
    public void setEta(double e) { eta = e; } public double getEta() {return eta; }

    private double lambda = 1.0;
    @OptionMetadata(displayName = "lambda", description = "lambda",
            commandLineParamName = "lambda", commandLineParamSynopsis = "-lambda <double>", displayOrder = 2)
    public void setLambda(double l) { lambda = l; } public double getLambda() {return lambda; }

    private double gamma = 1.0;
    @OptionMetadata(displayName = "gamma", description = "gamma",
            commandLineParamName = "gamma", commandLineParamSynopsis = "-gamma <double>", displayOrder = 3)
    public void setGamma(double l) { gamma = l; } public double getGamma() {return gamma; }

    private double subsample = 0.5;
    @OptionMetadata(displayName = "subsample", description = "subsample",
            commandLineParamName = "subsample", commandLineParamSynopsis = "-subsample <double>", displayOrder = 4)
    public void setSubsample(double s) { subsample = s; } public double getSubsample() {return subsample; }

    private double colsample_bynode = 1.0;
    @OptionMetadata(displayName = "colsample_bynode", description = "colsample_bynode",
            commandLineParamName = "colsample_bynode", commandLineParamSynopsis = "-colsample_bynode <double>", displayOrder = 5)
    public void setColSampleByNode(double c) { colsample_bynode = c; } public double getColSampleByNode() {return colsample_bynode; }

    private int max_depth = 6;
    @OptionMetadata(displayName = "max_depth", description = "max_depth",
            commandLineParamName = "max_depth", commandLineParamSynopsis = "-max_depth <int>", displayOrder = 6)
    public void setMaxDepth(int m) { max_depth = m; } public int getMaxDepth() {return max_depth; }

    private double min_child_weight = 1.0;
    @OptionMetadata(displayName = "min_child_weight", description = "min_child_weight",
            commandLineParamName = "min_child_weight", commandLineParamSynopsis = "-min_child_weight <double>", displayOrder = 7)
    public void setMinChildWeight(double w) { min_child_weight = w; } public double getMinChildWeight() {return min_child_weight; }
    private int min_instances_in_leaf = 1;
    // Default value is 1 - created one more hyper parameter by sumeet to determine min instance present at leaf
    @OptionMetadata(displayName = "min_instances_in_leaf", description = "Minimum instances required to split a node",
            commandLineParamName = "min_instances_in_leaf", commandLineParamSynopsis = "-min_instances_in_leaf <int>", displayOrder = 8)
    public void setMinInstancesInLeaf(int min_instances_in_leaf) {
        this.min_instances_in_leaf = min_instances_in_leaf;
    }

    public int getMinInstancesInLeaf() {
        return min_instances_in_leaf;
    }
    /** The main method for running this classifier from a command-line interface. */
    public static void main(String[] options) {
        runClassifier(new XGBoostTree(), options);
    }
}

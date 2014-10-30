import java.util.HashMap;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four
 * methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */

	private final double delta = 0.00001;
	private int vocabularySize;
	private int documentsCount;
	private int documentPerLabelCount[];
	private int wordPerLabelCount[];
	private Map<String, int[]> labelPerWordCount;

	public NaiveBayesClassifierImpl() {
		documentPerLabelCount = new int[Label.values().length];
		wordPerLabelCount = new int[Label.values().length];
		labelPerWordCount = new HashMap<String, int[]>();
	}

	@Override
	public void train(Instance[] trainingData, int v) {
		this.vocabularySize = v;
		this.documentsCount = trainingData.length;

		for (Instance instance : trainingData) {
			int label_ordinal = instance.label.ordinal();

			documentPerLabelCount[label_ordinal]++;
			wordPerLabelCount[label_ordinal] = wordPerLabelCount[label_ordinal]
					+ instance.words.length;

			for (String word : instance.words) {

				int[] labelCount = labelPerWordCount.get(word);
				if (labelCount == null) {
					labelCount = new int[Label.values().length];
					labelPerWordCount.put(word, labelCount);
				}
				labelCount[label_ordinal]++;
			}
		}
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or
	 * P(HAM)
	 */
	@Override
	public double p_l(Label label) {
		return (double) documentPerLabelCount[label.ordinal()] / documentsCount;
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		int count_w = 0;
		if (labelPerWordCount.containsKey(word))
			count_w = labelPerWordCount.get(word)[label.ordinal()];
		return (((double) count_w + delta) / ((vocabularySize * delta) + wordPerLabelCount[label
				.ordinal()]));
	}

	/**
	 * Classifies an array of words as either SPAM or HAM.
	 */
	@Override
	public ClassifyResult classify(String[] words) {

		double result[] = new double[Label.values().length];
		for (Label label : Label.values()) {
			double sum = Math.log(p_l(label));
			for (String word : words) {
				sum = sum
						+ Math.log(p_w_given_l(word, label));
			}
			result[label.ordinal()] = sum;
		}

		ClassifyResult classifyResult = new ClassifyResult();
		classifyResult.log_prob_ham = result[Label.HAM.ordinal()];
		classifyResult.log_prob_spam = result[Label.SPAM.ordinal()];
		classifyResult.label = (classifyResult.log_prob_spam > classifyResult.log_prob_ham) ? Label.SPAM
				: Label.HAM;
		return classifyResult;
	}
}

package cs107KNN;

import java.util.Arrays;
import java.util.Scanner;


public class KNN {
	public static void main(String[] args) {
		while(true) {
			System.out.println("=== Test predictions ===");
			Scanner scanner = new Scanner(System.in);
			System.out.println("Entrez la taille du dataset");
			String nombreimages = scanner.nextLine();
			long start = System.currentTimeMillis();

			byte[][][] imagesTrain = KNN.parseIDXimages(Helpers.readBinaryFile("mp/datasets/" + nombreimages + "-per-digit_images_train"));
			byte[] labelsTrain = KNN.parseIDXlabels(Helpers.readBinaryFile("mp/datasets/" + nombreimages + "-per-digit_labels_train"));

			byte[][][] imagesTest = KNN.parseIDXimages(Helpers.readBinaryFile("mp/datasets/10k_images_test"));
			byte[] labelsTest = KNN.parseIDXlabels(Helpers.readBinaryFile("mp/datasets/10k_labels_test"));

			int TESTS = 1000;

			byte[] predictions = new byte[TESTS];
			for (int i = 0; i < TESTS; i++) {
				predictions[i] = KNN.knnClassify(imagesTest[i], imagesTrain, labelsTrain, 7);
			}

			double accuracyTest = accuracy(predictions, labelsTest);

			long end = System.currentTimeMillis();
			double time = (end - start) / 1000d;
			System.out.println("Accuracy = " + accuracy(predictions, (Arrays.copyOfRange(labelsTest, 0, TESTS))) * 100 + " %");
			System.out.println("Time = " + time + " seconds");
			System.out.println("Time per test image = " + (time / TESTS));
		}
	}
	
	public static Scanner keyb = new Scanner(System.in);
	
	/**
	 * Composes four bytes into an integer using big endian convention.
	 *
	 * @param bXToBY The byte containing the bits to store between positions X and Y
	 * 
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {

		byte[] list = {b31ToB24, b23ToB16, b15ToB8, b7ToB0};
		
		String sumString = "";
		int sumInt = 0;
		
		for (int j = 0; j < list.length; j++) {
			sumString += Helpers.byteToBinaryString(list[j]);
		}
				
		sumInt = Integer.parseInt(sumString, 2);
		
		return sumInt;
	}

	/**
	 * Parses an IDX file containing images
	 *
	 * @param data the binary content of the file
	 *
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXimages(byte[] data) {
		
        int numberImages = extractInt(data[4],data[5],data[6],data[7]);
        int HeightImages = extractInt(data[8],data[9],data[10],data[11]);
        int WidthImages = extractInt(data[12],data[13],data[14],data[15]);
        
        int numberPixelsPerImage = HeightImages * WidthImages;

        byte pixelValue;
        
        byte[][][] arrayTensor = new byte[numberImages][HeightImages][WidthImages];

        for(int c = 0; c < numberImages ; c++) {
        	
           for(int b = 0; b<HeightImages; b++) {
        	   
              for (int i = 0; i < WidthImages; i++) {
            	  
            	 pixelValue =  (byte) ((data[i + b*WidthImages + c*numberPixelsPerImage + 15]) - 128) ;
                 arrayTensor[c][b][i] = pixelValue;
                 
              }

           }

        }

        return arrayTensor;
	}
	
	

	/**
	 * Parses an idx images containing labels
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static byte[] parseIDXlabels(byte[] data) {
		byte[] labels = new byte[extractInt(data[4], data[5], data[6],data[7])];
		
		int i = 0;
		
		for (int k = 8; k < labels.length; k++) {
			labels[i] = data[k];
			i++;
		}
		
		return labels;
	}

	/**
	 * @brief Computes the squared L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {
		float sum = 0;
		
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				sum +=  ((a[i][j] - b[i][j]) * (a[i][j] - b[i][j]));
			}

		}


		return sum;
	}

	/**
	 * @brief Computes the inverted similarity between 2 images.
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the inverted similarity between the two images
	 */
	public static float invertedSimilarity(byte[][] a, byte[][] b) {
		float simInv = 0;
		float sum1 = 0, sum2 = 0, sum3 = 0, racine;
		float[] moyennes = moyenne(a,b);
		
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				sum1 += ((a[i][j] - moyennes[0]) * (b[i][j] - moyennes[1]));
				sum2 += ((a[i][j] - moyennes[0]) * (a[i][j] - moyennes[0]));
				sum3 += ((b[i][j] - moyennes[1]) * (b[i][j] - moyennes[1]));
			}
		}
		
		racine = (float) Math.sqrt(sum2 * sum3);
		simInv = 1 - (sum1 / racine);
		
		return simInv;
	}
	
	/**
	 * @brief Computes the average for two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return size two array of both averages
	 */
	public static float[] moyenne(byte[][] a, byte[][] b) {
		float[] moyenne = new float[2];
		
		float produit = a.length * a[0].length;
		
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				moyenne[0] += a[i][j];
				moyenne[1] += b[i][j];
			}
		}
		
		moyenne[0] /= produit;
		moyenne[1] /= produit;
		
		return moyenne;
	}

	/**
	 * @brief Quicksorts and returns the new indices of each value.
	 * 
	 * @param values the values whose indices have to be sorted in non decreasing
	 *               order
	 * 
	 * @return the array of sorted indices
	 * 
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */
	public static int[] quicksortIndices(float[] values) {
		int[] indices = new int[values.length];
		
		for (int i = 0; i < values.length; i++) {
			indices[i] = i;
		}
		
		int l = 0;
		int h = values.length - 1;
		
		quicksortIndices(values, indices, l, h);
		
		return indices;
	}

	/**
	 * @brief Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 * 
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {
		float pivot = values[low];
		int l = low;
		int h = high;
		
		while (l <= h) {
			if (values[l] < pivot) {
				l++;
			} else if (values[h] > pivot) {
				h--;
			} else {
				swap(l, h, values, indices);
				l++;
				h--;
			}
		}
		
		if (low < h) {
			quicksortIndices(values, indices, low, h);
		}
		
		if (high > l) {
			quicksortIndices(values, indices, l, high);
		}
	
	}

	/**
	 * @brief Swaps the elements of the given arrays at the provided positions
	 * 
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {
		float temp;
		int temp2;
		
		temp = values[i];
		values[i] = values[j];
		values[j] = temp;
		
		temp2 = indices[i];
		indices[i] = indices[j];
		indices[j] = temp2;
	}

	/**
	 * @brief Returns the index of the largest element in the array
	 * 
	 * @param array an array of integers
	 * 
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {
			int maxElement = array[0];
			int index = 0;

			for(int i =0;i<array.length; i++){
				if(array[i] > maxElement){
					maxElement = array[i];
					index = i;
				}
			}


		return index;
	}

	/**
	 * The k first elements of the provided array vote for a label
	 *
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 *
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {

		int[] tab = new int[10];
		
		// On itère les k labels les plus proches, c'est à dire les k premiers indices.
		
		for(int i=0; i<k; i++){
			tab[labels[sortedIndices[i]]] +=1;
		}

		return (byte)indexOfMax(tab);
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 *
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 *
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {		
		float[] distances = new float[trainImages.length];
		
		
		int choix = 2;
		
		for (int i = 0; i < trainImages.length; i++) {
			distances[i] = (choix == 1) ? squaredEuclideanDistance(image, trainImages[i]) : invertedSimilarity(image, trainImages[i]);
		}
		
		int[] indices = quicksortIndices(distances);
		
		int label = electLabel(indices, trainLabels, k);
		
		return (byte)label;
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
		// TODO: Implémenter
		double sum = 0;

		for(int i =0; i<predictedLabels.length; i++){

			sum += (predictedLabels[i] == trueLabels[i]) ? 1 : 0;

		}
		double accuracy = (sum/predictedLabels.length);

		return accuracy;
	}
}

package cs107KNN;

import java.sql.SQLOutput;
import java.util.Set;
import java.util.HashSet;
import java.util.Random;
import java.util.ArrayList;

public class KMeansClustering {
	public static void main(String[] args) {
		int K = 5000;
		int maxIters = 20;

		 //TODO: Adaptez les parcours
		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("mp/datasets/1000-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("mp/datasets/1000-per-digit_labels_train"));

		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);

		byte[] reducedLabels = new byte[reducedImages.length];
		for (int i = 0; i < reducedLabels.length; i++) {
			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
		}



		byte[] list = new byte[4];



		//Helpers.writeBinaryFile("datasets/reduced10Kto1K_images", encodeIDXimages(reducedImages));
		//Helpers.writeBinaryFile("datasets/reduced10Kto1K_labels", encodeIDXlabels(reducedLabels));


		encodeInt(2051,list,0);

		for(int i = 0; i<4; i++){
			System.out.println(list[i]);
		}
	}

    /**
     * @brief Encodes a tensor of images into an array of data ready to be written on a file
     * 
     * @param images the tensor of image to encode
     * 
     * @return the array of byte ready to be written to an IDX file
     */
	public static byte[] encodeIDXimages(byte[][][] images) {
		byte[] data = new byte[images.length * images[0].length * images[0][0].length];
		
		encodeInt(2051, data, 0);
		encodeInt(images.length, data, 4);
		encodeInt(images[0].length, data, 8);
		encodeInt(images[0][0].length, data, 12);
		
		for (int i = 16; i < images.length; i++) {
			for (int j = 0; j < images[0].length; j++) {
				for (int k = 0; k < images[0][0].length; k++) {
					data[i+j+k] = images[i][j][k];
				}
			}
		}
		
		return data;
	}

    /**
     * @brief Prepares the array of labels to be written on a binary file
     * 
     * @param labels the array of labels to encode
     * 
     * @return the array of bytes ready to be written to an IDX file
     */
	public static byte[] encodeIDXlabels(byte[] labels) {
		byte[] data = new byte[labels.length + 4];
		
		encodeInt(2049, data, 0);
		encodeInt(labels.length, data, 4);
		
		for (int i = 4; i < data.length; i++) {
			data[i] = labels[i-4];
		}
		
		return data;
	}

    /**
     * @brief Decomposes an integer into 4 bytes stored consecutively in the destination
     * array starting at position offset
     * 
     * @param n the integer number to encode
     * @param destination the array where to write the encoded int
     * @param offset the position where to store the most significant byte of the integer,
     * the others will follow at offset + 1, offset + 2, offset + 3
     */
	public static void encodeInt(int n, byte[] destination, int offset) {
		// TODO: Implémenter
		char[] tempList = new char[32];
		String ourByte  = Integer.toBinaryString(n);
		int length = ourByte.length();

		for(int i=0;i<32; i++){
			if (32-(length+i)>0){
				tempList[i] = '0';

			}
			else{
				tempList[i] = ourByte.charAt(i-(32-length));

			}
		}

		String convertedByte = new String(tempList);


		destination[offset] = (byte)Integer.parseInt(convertedByte.substring(0,8),2);
		destination[offset + 1 ] = (byte)Integer.parseInt(convertedByte.substring(8,16),2);
		destination[offset + 2 ] = (byte)Integer.parseInt(convertedByte.substring(16,24),2);
		destination[offset + 3 ] = (byte)Integer.parseInt(convertedByte.substring(24,32),2);
	}

    /**
     * @brief Runs the KMeans algorithm on the provided tensor to return size elements.
     * 
     * @param tensor the tensor of images to reduce
     * @param size the number of images in the reduced dataset
     * @param maxIters the number of iterations of the KMeans algorithm to perform
     * 
     * @return the tensor containing the reduced dataset
     */
	public static byte[][][] KMeansReduce(byte[][][] tensor, int size, int maxIters) {
		int[] assignments = new Random().ints(tensor.length, 0, size).toArray();
		byte[][][] centroids = new byte[size][][];
		initialize(tensor, assignments, centroids);

		int nIter = 0;
		while (nIter < maxIters) {
			// Step 1: Assign points to closest centroid
			recomputeAssignments(tensor, centroids, assignments);
			System.out.println("Recomputed assignments");
			// Step 2: Recompute centroids as average of points
			recomputeCentroids(tensor, centroids, assignments);
			System.out.println("Recomputed centroids");

			System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);

			nIter++;
		}

		return centroids;
	}

   /**
     * @brief Assigns each image to the cluster whose centroid is the closest.
     * It modifies.
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void recomputeAssignments(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
		int numberCentroids = centroids.length;
		int tensorLength = tensor.length;

		//Iterate through the tensor and calculate the distance with each centroids.
		for(int i = 0; i<tensorLength; i++){
			float[] distancesForICluster = new float[numberCentroids];

			for(int b = 0; b < numberCentroids; b++){

				distancesForICluster[b] = KNN.squaredEuclideanDistance(tensor[i], centroids[b]);
			}

			int closestCluster = indexOfMaxFloat(distancesForICluster);
			//On place le numéro du cluster à la position correspondante de l'image i dans assignments.
			assignments[i] = closestCluster;
			System.out.println(closestCluster);
		}

	}

	public static int indexOfMaxFloat(float[] array) {
		float maxElement = array[0];
		int index = 0;

		for(int i =0;i<array.length; i++){
			if(array[i] > maxElement){
				maxElement = array[i];
				index = i;
			}
		}

		if(index ==1){
			System.out.println("hey");
		}

		return index;
	}

    /**
     * @brief Computes the centroid of each cluster by averaging the images in the cluster
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void recomputeCentroids(byte[][][] tensor, byte[][][] centroids, int[] assignments) {



		return;



	}


    /**
     * Initializes the centroids and assignments for the algorithm.
     * The assignments are initialized randomly and the centroids
     * are initialized by randomly choosing images in the tensor.
     * 
     * @param tensor the tensor of images to cluster
     * @param assignments the vector indicating to what cluster each image belongs to.
     * @param centroids the tensor of centroids that represent the cluster of images
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {
		Set<Integer> centroidIds = new HashSet<>();
		Random r = new Random("cs107-2018".hashCode());
		while (centroidIds.size() != centroids.length)
			centroidIds.add(r.nextInt(tensor.length));
		Integer[] cids = centroidIds.toArray(new Integer[] {});
		for (int i = 0; i < centroids.length; i++)
			centroids[i] = tensor[cids[i]];
		for (int i = 0; i < assignments.length; i++)
			assignments[i] = cids[r.nextInt(cids.length)];
	}
}

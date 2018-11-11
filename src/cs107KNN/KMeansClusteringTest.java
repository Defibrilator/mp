package cs107KNN;

import java.util.Arrays;

public class KMeansClusteringTest {


    public static void main(String[] args){

        encodeIDXImagesTest();
        recomputeAssignmentsTest();
        recomputeCentroidsTest();

    }

    public static void encodeIDXImagesTest(){

        byte[] originalImages = Helpers.readBinaryFile("mp/datasets/10-per-digit_images_train");
        byte[] originalLabels = Helpers.readBinaryFile("mp/datasets/10-per-digit_labels_train");

        byte[][][] imagesTrain = KNN.parseIDXimages(originalImages);
        byte[] labelsTrain = KNN.parseIDXlabels(originalLabels);

        byte [] testImages = KMeansClustering.encodeIDXimages(imagesTrain);
        byte[] testLabels  = KMeansClustering.encodeIDXlabels(labelsTrain);

        byte original = -40;
        int convertInt = (original & 0xFF);
        byte converted = (byte)(convertInt-128);

        for(int i=0; i<testImages.length; i++){
            if(originalImages[i] != testImages[i]){
                System.out.println("Erreur rencontrée en indice " + i );
                System.out.println("Fichier original:" + originalImages[i]);
                System.out.println("Fichier reconvertit:" + testImages[i]);

            }
        }

        System.out.println("Comparison before and after IMAGES: " + java.util.Arrays.equals(testImages,originalImages));
        System.out.println("Comparison before and after LABELS: " + java.util.Arrays.equals(testLabels,originalLabels));



    }


    public static void recomputeAssignmentsTest(){

        byte[][][] tensorTest = new byte[][][] {{{1,0},{1,1}},{{1, 1}, {2, 2}}};
        byte[][][] centroidsTest = new byte[][][] {{{1,1},{1,2}},{{3,4},{4,3}}};


        int[] expectedAssignments = new int[]{0,0};
        int[] resultAssignments = new int[2];
        KMeansClustering.recomputeAssignments(tensorTest,centroidsTest, resultAssignments);

        //KMeansClustering.recomputeAssignments();

        System.out.println("Expected results: " + Arrays.toString(expectedAssignments));
        System.out.println("Results: " + Arrays.toString(resultAssignments));



    }


    public static void recomputeCentroidsTest(){

        byte[][][] tensorTest = new byte[][][] {{{1,0},{1,1}},{{1, 1}, {2, 2}}};
        byte[][][] centroidsTest = new byte[][][] {{{1,1},{1,2}},{{3,4},{4,3}}};

        int[] assignments = new int[]{0,0};

        byte[][][] centroidsExpected = new byte[][][] {{{1,1/2},
                {1,1}},
                {{3,4},
                        {4,3}}};

        KMeansClustering.recomputeCentroids(tensorTest,centroidsTest, assignments);
        System.out.println("Centroids expected");
        System.out.println(Arrays.toString(centroidsExpected[0][0]));
        System.out.println(Arrays.toString(centroidsExpected[0][1]));
        System.out.println(Arrays.toString(centroidsExpected[1][0]));
        System.out.println(Arrays.toString(centroidsExpected[1][1]));

        System.out.println("Centroids results");
        System.out.println(Arrays.toString(centroidsTest[0][0]));
        System.out.println(Arrays.toString(centroidsTest[0][1]));
        System.out.println(Arrays.toString(centroidsTest[1][0]));
        System.out.println(Arrays.toString(centroidsTest[1][1]));




    }




}

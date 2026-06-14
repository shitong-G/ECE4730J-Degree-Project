/******************************************************************************
* 
* arr_search.c
*
* This program implements several library function,
* and can be used to illustrate differences between static and dynamic linking.
* It can also be used as a hypothetical workload.
*
* Each iteration of the program's work function
* allocates an array of floats using malloc()
* The array is filled by calculating the square root, with sqrt(),
* of the array's index.
* Then, bsearch() is used to find a value in the array.
* The array is cleared with memset().
* Finally, free() is called to free the allocated memory.
*
* Usage: This program takes a single input describing the number of
*        iterations to run.
*
* Written August 14, 2020 by Marion Sudvarg
******************************************************************************/

#include <stdio.h> //For printf
#include <stdlib.h> //For atoi, malloc, free, bsearch
#include <math.h> //For sqrt
#include <string.h> //For memset

#define ARR_SIZE 512
#define ARG_ITERATIONS 1
#define NUM_ARGS ( ARG_ITERATIONS + 1 )

//Compare floating point values in bsearch
int compare_float(const void * f1, const void * f2) {
        return ( *( (float *) f1) - *( (float*) f2) );
}

//Workload for each iteration
int library_calls(void) {

    float * values, * value;
    float key;
    int i;

    //Allocate float array
    values = (float *) malloc(ARR_SIZE * sizeof(float));
    if(!values) return -1;

    //Assign values to array with sqrt()
    for (i=0; i<ARR_SIZE; i++) {
            values[i] = sqrt(i+1);
    }

    //Find value in array
    key = sqrt(383);
    value = (float *) bsearch (&key, values, ARR_SIZE, sizeof(float), compare_float);

    //Clear array memory with memset()
    memset(values, 0, ARR_SIZE * sizeof(float));

    //Free array memory
    free(values);

    return 0;

}

int main (int argc, char * argv[]) {

    int i, iterations;

    //Make sure iterations are specified
    if (argc < NUM_ARGS) {
        printf("Usage: %s <iterations>\n", argv[0]);
        return -1;
    }

    iterations = atoi(argv[ARG_ITERATIONS]);

    //Specified iterations must be positive
    if (iterations <= 0) {
        printf("ERROR: Iteration count must be greater than 0!\n");
        return -1;
    }

    //Execute workload for specified iterations
    for (int i = 0; i < iterations; i++) {
        //Break if allocation fails
        if (library_calls()) return -1;          
    }

    printf("%s completed %d iterations\n", argv[0], iterations);
    
    return 0;

}

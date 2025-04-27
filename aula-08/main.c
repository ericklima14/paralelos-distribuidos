#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

int *scan(int *A, int size){
    int *B = (int *) malloc(sizeof(int) * (size/2));
    int *P = (int *) malloc(sizeof(int) * size);

    P[0] = A[0];

    if(size > 1){
        int metade = size/2;

        #pragma omp parallel for
        for(int i = 0; i < metade; i++){
            B[i] = A[2 * i] + A[2 * i + 1];
        }
        int *C = scan(B, metade);

        #pragma omp parallel for
        for(int i = 1; i < size; i++){
            if(i % 2 == 0){
                P[i] = C[i/2];
            }
            else {
                P[i] = C[(i - 1)/2] + A[i];
            }
        }

        free(B);
        free(C);
    }

    return P;
}

int* filter(int *A, int *flags, int size){
    flags = scan(flags, size);

    int *B = (int *) malloc(sizeof(int) * (flags[size - 1] + 1));

    #pragma omp parallel for
    for(int i = 1; i < size; i++){
        if(flags[i] != flags[i-1]){
            B[flags[i]] = A[i];
        }
    }

    if(flags[0] == 1){
        B[0] = A[0];
    }

    return B;
}

void broadcast(int x, int *A, int inicio, int final){
    if (inicio == final){
        A[inicio] = x;
    } else{
        int meio = (inicio + final)/2;
        int meio2 = meio + 1;
        int x2 = x;

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                broadcast(x, A, inicio, meio);
                
                #pragma omp task
                broadcast(x2, A, meio2, final);
            }
        }
    }
}

int *Mark(int *A, int size){
    int *flags = (int *) malloc(sizeof(int) * size);
    int *pivos = (int *) malloc(sizeof(int) * size);
    broadcast(A[0], pivos, 0, size - 1);

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        if(pivos[i] <= A[i]){
            flags[i] = 1;
        } else {
            flags[i] = 0;
        }
    }

    return flags;
}

void combine(int *B, int *C, int *A, int k, int m){
    #pragma omp parallel for
    for(int i = 0; i < k; i++){
        A[i] = B[i];
    }

    #pragma omp parallel for
    for(int i = 0; i < m; i++){
        A[k + i] = C[i];
    }
}

void swap(int *a, int *b){
    int aux = *b;
    *b = *a;
    *a = aux;
    return;
}

void flip(int *flags, int size){
    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        flags[i] = !flags[i];
    }
}

int partition(int *A, int size){
    int *flags = Mark(A, size);
    int *B = filter(A, flags, size);
    int k = flags[size - 1];
    
    flip(flags, size);
    
    int *C = filter(A, flags, size);
    int m = size - k;

    combine(B, C, A, k, m);
    swap(&A[0], &A[k]);
    return k;
}

void quicksort(int *A, int i, int j){
    if (i < j){
        int pivot = (rand() % (j - i + 1)) + i;
        swap(&A[i], &A[pivot]);
        int k = partition(A + i, j - i + 1);
        k += i;

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                quicksort(A, i, k - 1);

                #pragma omp task
                quicksort(A, k + 1, j);
            }
        }
    }
}


int* compact(int **arrays, int *sizes, int size_of_sizes){
    sizes[0] = 0;
    sizes = scan(sizes, size_of_sizes);
    int *A = (int *) malloc(sizeof(int) * sizes[size_of_sizes]);

    #pragma omp parallel for
    for(int i = 0; i < size_of_sizes; i++){
        int start = sizes[i];
        int len = sizes[i + 1] - sizes[i];

        for(int j = 0; j < len; j++){
            A[start + j] = arrays[i][j];
        }
    }

    return A;
}

int main(){
    srand(time(NULL));

    int arr[] = {9, 2, 7, 4, 5, 8, 1, 6, 3};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Array antes do quicksort:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    quicksort(arr, 0, n - 1);

    printf("Array depois do quicksort:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n\n");

    int size_of_sizes = 3;
    int sizes[] = {0, 2, 3, 1};

    int *arrays[3];
    arrays[0] = (int *)malloc(2 * sizeof(int));
    arrays[1] = (int *)malloc(3 * sizeof(int));
    arrays[2] = (int *)malloc(1 * sizeof(int));

    arrays[0][0] = 1;
    arrays[0][1] = 2;

    arrays[1][0] = 3;
    arrays[1][1] = 4;
    arrays[1][2] = 5;

    arrays[2][0] = 6;

    int *compacted = compact(arrays, sizes, size_of_sizes);

    printf("Array depois do compact:\n");
    for (int i = 0; i < sizes[size_of_sizes]; i++) {
        printf("%d ", compacted[i]);
    }
    printf("\n");

    for (int i = 0; i < size_of_sizes; i++) {
        free(arrays[i]);
    }
    free(compacted);

    return 0;
}
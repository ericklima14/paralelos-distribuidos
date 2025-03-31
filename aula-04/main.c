#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>

int* soma_de_prefixos(int* A, int n) {
    int *B = (int*)malloc(sizeof(int) * (2 * n));
    int *P = (int*)malloc(sizeof(int) * (2 * n));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        B[n + i] = A[i];
    }

    int logN = (int)log2(n);
    for (int j = logN - 1; j >= 0; j--) {
        int expon2 = 1 << j;
        int expon2_1   = 1 << (j + 1);

        #pragma omp parallel for
        for (int i = expon2; i < expon2_1; i++) {
            B[i] = B[2*i] + B[2*i + 1];
        }
    }

    P[1] = 0;

    for (int j = 0; j < logN; j++) {
        int expon2 = 1 << j;
        int expon2_1 = 1 << (j + 1);

        #pragma omp parallel for
        for (int i = expon2; i < expon2_1; i++) {
            int parte_esq  = 2 * i;
            int parte_dir = parte_esq + 1;
            if (parte_esq < 2 * n) {
                P[parte_esq] = P[i];
            }
            if (parte_dir < 2 * n) {
                P[parte_dir] = P[i] + B[parte_esq];
            }
        }
    }

    int *resultado = (int*)malloc(sizeof(int) * n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        resultado[i] = P[n + i] + A[i];
    }

    free(B);
    free(P);

    return resultado;
}

void prefix_sum(int *A, int i, int j){
    if(i >= j)
        return;
    
    int meio = (int) (i+j)/2;

    #pragma omp parallel shared(A, i, j, meio)
    {
        prefix_sum(A, i, meio);
        prefix_sum(A, meio + 1, j);
        
        #pragma omp for
        for(int k = meio + 1; k < j; k++){
            A[k] = A[k] + A[meio];
        }
    }
}

int *prefix_sum_efficient(int *A, int size){    
    int *P = (int *) malloc(size * sizeof(int));
    
    if (size == 1) {
        P[0] = A[0];
    }
    else {
        int *B = (int *) malloc((size/2) * sizeof(int));

        #pragma omp parallel for
        for(int i = 0; i < size/2; i++){
            B[i] = A[2* i] + A[2 * i + 1];
        }

        int *C = prefix_sum_efficient(B, size/2);
        free(B);

        P[0] = A[0];

        #pragma omp parallel for
        for(int i = 1; i < size; i++){
            if(i % 2 == 1)
                P[i] = C[i/2];
            else
                P[i] = C[(i-1)/2] + A[i];
        }
        free(C);
    }
    
    return P;
}

void print_array(int *A, int size){
    for(int i = 0; i < size; i++){
        printf("%d ", A[i]);
    }
    printf("\n===========================\n");
}

int main(){
    int A[4] = {1, 2, 3, 4};
    int *result = soma_de_prefixos(A, 4);
    print_array(result, 4);
    result = prefix_sum_efficient(A, 4);
    print_array(result, 4);

    return 0;
}
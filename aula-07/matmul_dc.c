// matmul_dc.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define THRESHOLD 64  // tamanho mínimo para parar a recursão

// sub-bloco de C += A * B
void matmul_dc_rec(double *A, double *B, double *C, int n,
                   int aRow, int aCol, int bRow, int bCol, int cRow, int cCol, int size)
{
    if (size <= THRESHOLD) {
        // caso base: multiplicação ingênua
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                double sum = 0;
                for (int k = 0; k < size; k++)
                    sum += A[(aRow + i)*n + (aCol + k)]
                         * B[(bRow + k)*n + (bCol + j)];
                C[(cRow + i)*n + (cCol + j)] += sum;
            }
    } else {
        int m = size / 2;
        // 8 sub-produtos para cada quadrante de C
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow,     aCol,     bRow,     bCol,     cRow,     cCol,     m);
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow,     aCol + m, bRow + m, bCol,     cRow,     cCol,     m);
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow,     aCol,     bRow,     bCol + m, cRow,     cCol + m, m);
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow,     aCol + m, bRow + m, bCol + m, cRow,     cCol + m, m);
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow + m, aCol,     bRow,     bCol,     cRow + m, cCol,     m);
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow + m, aCol + m, bRow + m, bCol,     cRow + m, cCol,     m);
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow + m, aCol,     bRow,     bCol + m, cRow + m, cCol + m, m);
        #pragma omp task shared(A,B,C)
        matmul_dc_rec(A,B,C,n,aRow + m, aCol + m, bRow + m, bCol + m, cRow + m, cCol + m, m);
        #pragma omp taskwait
    }
}

void matmul_dc(double *A, double *B, double *C, int n)
{
    // inicializa C com zeros
    for (int i = 0; i < n*n; i++) C[i] = 0.0;
    #pragma omp parallel
    {
        #pragma omp single
        matmul_dc_rec(A, B, C, n, 0,0, 0,0, 0,0, n);
    }
}

int main(int argc, char **argv)
{
    int n = 1024;  // deve ser potência de 2
    double *A = malloc(n*n*sizeof(double));
    double *B = malloc(n*n*sizeof(double));
    double *C = malloc(n*n*sizeof(double));

    // preenche A e B com valores aleatórios
    for (int i = 0; i < n*n; i++) {
        A[i] = drand48();
        B[i] = drand48();
    }

    double t0 = omp_get_wtime();
    matmul_dc(A, B, C, n);
    double t1 = omp_get_wtime();
    printf("MatMul DC: %f s\n", t1 - t0);

    free(A); free(B); free(C);
    return 0;
}
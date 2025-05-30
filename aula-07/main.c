// matmul_strassen.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define STRASSEN_THRESHOLD 64

// C = A + B
void add(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n*n; i++) C[i] = A[i] + B[i];
}
// C = A - B
void sub(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n*n; i++) C[i] = A[i] - B[i];
}

void strassen_rec(double *A, double *B, double *C, int n)
{
    if (n <= STRASSEN_THRESHOLD) {
        // fallback para produto ingênuo
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++)
                    sum += A[i*n+k] * B[k*n+j];
                C[i*n+j] = sum;
            }
        return;
    }

    int m = n/2, sz = m*m;
    // dividindo A, B e C em 4 sub-matrizes cada
    double *A11 = A;
    double *A12 = A + m;
    double *A21 = A + n*m;
    double *A22 = A + n*m + m;
    double *B11 = B;
    double *B12 = B + m;
    double *B21 = B + n*m;
    double *B22 = B + n*m + m;
    double *C11 = C;
    double *C12 = C + m;
    double *C21 = C + n*m;
    double *C22 = C + n*m + m;

    // aloca temporárias
    double *S1  = malloc(sz*sizeof(double));
    double *S2  = malloc(sz*sizeof(double));
    double *S3  = malloc(sz*sizeof(double));
    double *S4  = malloc(sz*sizeof(double));
    double *S5  = malloc(sz*sizeof(double));
    double *S6  = malloc(sz*sizeof(double));
    double *S7  = malloc(sz*sizeof(double));
    double *S8  = malloc(sz*sizeof(double));
    double *S9  = malloc(sz*sizeof(double));
    double *S10 = malloc(sz*sizeof(double));
    double *P1  = malloc(sz*sizeof(double));
    double *P2  = malloc(sz*sizeof(double));
    double *P3  = malloc(sz*sizeof(double));
    double *P4  = malloc(sz*sizeof(double));
    double *P5  = malloc(sz*sizeof(double));
    double *P6  = malloc(sz*sizeof(double));
    double *P7  = malloc(sz*sizeof(double));

    // calcula S1..S10
    sub(B12, B22, S1,  m);  // S1 = B12 - B22
    add(A11, A12, S2,  m);  // S2 = A11 + A12
    add(A21, A22, S3,  m);  // S3 = A21 + A22
    sub(B21, B11, S4,  m);  // S4 = B21 - B11
    add(A11, A22, S5,  m);  // S5 = A11 + A22
    add(B11, B22, S6,  m);  // S6 = B11 + B22
    sub(A12, A22, S7,  m);  // S7 = A12 - A22
    add(B21, B22, S8,  m);  // S8 = B21 + B22
    sub(A11, A21, S9,  m);  // S9 = A11 - A21
    add(B11, B12, S10, m);  // S10 = B11 + B12

    // calcula P1..P7 em tarefas paralelas
    #pragma omp task shared(A11,S1,P1)  strassen_rec(A11,  S1,  P1,  m);
    #pragma omp task shared(S2,B22,P2)  strassen_rec(S2,   B22, P2,  m);
    #pragma omp task shared(S3,B11,P3)  strassen_rec(S3,   B11, P3,  m);
    #pragma omp task shared(A22,S4,P4)  strassen_rec(A22,  S4,  P4,  m);
    #pragma omp task shared(S5,S6,P5)   strassen_rec(S5,   S6,  P5,  m);
    #pragma omp task shared(S7,S8,P6)   strassen_rec(S7,   S8,  P6,  m);
    #pragma omp task shared(S9,S10,P7)  strassen_rec(S9,   S10, P7,  m);
    #pragma omp taskwait

    // monta C11..C22
    // C11 = P5 + P4 - P2 + P6
    for (int i = 0; i < sz; i++)
        C11[i] = P5[i] + P4[i] - P2[i] + P6[i];
    // C12 = P1 + P2
    for (int i = 0; i < sz; i++)
        C12[i] = P1[i] + P2[i];
    // C21 = P3 + P4
    for (int i = 0; i < sz; i++)
        C21[i] = P3[i] + P4[i];
    // C22 = P5 + P1 - P3 - P7
    for (int i = 0; i < sz; i++)
        C22[i] = P5[i] + P1[i] - P3[i] - P7[i];

    // libera tudo
    free(S1); free(S2); free(S3); free(S4); free(S5);
    free(S6); free(S7); free(S8); free(S9); free(S10);
    free(P1); free(P2); free(P3); free(P4); free(P5); free(P6); free(P7);
}

void strassen(double *A, double *B, double *C, int n)
{
    // inicializa C nulo
    for (int i = 0; i < n*n; i++) C[i] = 0.0;
    #pragma omp parallel
    {
        #pragma omp single
        strassen_rec(A,B,C,n);
    }
}

int main()
{
    int n = 1024;  // potência de 2
    double *A = malloc(n*n*sizeof(double));
    double *B = malloc(n*n*sizeof(double));
    double *C = malloc(n*n*sizeof(double));

    for (int i = 0; i < n*n; i++) {
        A[i] = drand48();
        B[i] = drand48();
    }

    double t0 = omp_get_wtime();
    strassen(A,B,C,n);
    double t1 = omp_get_wtime();
    printf("Strassen: %f s\n", t1 - t0);

    free(A); free(B); free(C);
    return 0;
}
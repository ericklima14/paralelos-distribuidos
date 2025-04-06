#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define N 4 
#define V 8 

void imprime_matriz(int **mat, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++)
            printf("%d ", mat[i][j]);
        printf("\n");
    }
    printf("\n");
}

void imprime_vetor(int *v, int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", v[i]);
    printf("\n\n");
}

void adicionar_matriz(int **C, int cLin, int cCol,
                      int **T, int tLin, int tCol, 
                      int size){
    if(size == 1)
        C[cLin][cCol] = C[cLin][cCol] + T[tLin][tCol];
    else{
        int metade = size/2;

        #pragma omp parallel
        {
            #pragma omp task
            adicionar_matriz(C, cLin, cCol, T, tLin, tCol, metade);

            #pragma omp task
            adicionar_matriz(C, cLin, cCol + metade, T, tLin, tCol + metade, metade);

            #pragma omp task
            adicionar_matriz(C, cLin + metade, cCol, T, tLin + metade, tCol, metade);

            #pragma omp task
            adicionar_matriz(C, cLin + metade, cCol + metade, T, tLin + metade, tCol + metade, metade);
        }
    }
}

void multi_matrizes(int **C, int cLin, int cCol,
                    int **A, int aLin, int aCol,
                    int **B, int bLin, int bCol,
                    int size){
    if(size == 1)
        C[cLin][cCol] = A[aLin][aCol] * B[bLin][bCol];
    else{
        int **T = (int **) malloc(size * sizeof(int*));
        for(int i = 0; i < size; i++){
            T[i] = (int *) malloc(size * sizeof(int));
        }

        int metade = size/2;

        #pragma omp parallel
        {
            #pragma omp single
            {
                #pragma omp task
                multi_matrizes(C, cLin, cCol, A, aLin, aCol, B, bLin, bCol, metade);
                #pragma omp task
                multi_matrizes(C, cLin, cCol + metade, A, aLin, aCol, B, bLin, bCol + metade, metade);
                #pragma omp task
                multi_matrizes(C, cLin + metade, cCol, A, aLin + metade, aCol, B, bLin, bCol, metade);
                #pragma omp task
                multi_matrizes(C, cLin + metade, cCol + metade, A, aLin + metade, aCol, B, bLin, bCol + metade, metade);
                
                #pragma omp task
                multi_matrizes(T, 0, 0, A, aLin, aCol + metade, B, bLin + metade, bCol, metade);
                #pragma omp task
                multi_matrizes(T, 0, metade, A, aLin, aCol + metade, B, bLin + metade, bCol + metade, metade);
                #pragma omp task
                multi_matrizes(T, metade, 0, A, aLin + metade, aCol + metade, B, bLin + metade, bCol, metade);
                #pragma omp task
                multi_matrizes(T, metade, metade, A, aLin + metade, aCol + metade, B, bLin + metade, bCol + metade, metade); 
                
                #pragma omp taskwait
                
                #pragma omp task
                adicionar_matriz(C, cLin, cCol, T, 0, 0, size);
                #pragma omp taskwait
            }
        }

        for (int i = 0; i < size; i++){
            free(T[i]);
        }
        free(T);
    }
}

int** alocaMatriz(int n) {
    int** matriz = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matriz[i] = (int*)calloc(n, sizeof(int));
    }
    return matriz;
}

void liberaMatriz(int** matriz, int n) {
    for (int i = 0; i < n; i++)
        free(matriz[i]);
    free(matriz);
}

void soma(int** A, int** B, int** C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void subtrai(int** A, int** B, int** C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void strassen(int **A, int **B, int **C, int n) {
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0]; 
        return;
    }

    int metade = n / 2;

    int** A11 = alocaMatriz(metade);
    int** A12 = alocaMatriz(metade);
    int** A21 = alocaMatriz(metade);
    int** A22 = alocaMatriz(metade);
    int** B11 = alocaMatriz(metade);
    int** B12 = alocaMatriz(metade);
    int** B21 = alocaMatriz(metade);
    int** B22 = alocaMatriz(metade);

    for (int i = 0; i < metade; i++) {
        for (int j = 0; j < metade; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + metade];
            A21[i][j] = A[i + metade][j];
            A22[i][j] = A[i + metade][j + metade];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + metade];
            B21[i][j] = B[i + metade][j];
            B22[i][j] = B[i + metade][j + metade];
        }
    }

    int** M1 = alocaMatriz(metade);
    int** M2 = alocaMatriz(metade);
    int** M3 = alocaMatriz(metade);
    int** M4 = alocaMatriz(metade);
    int** M5 = alocaMatriz(metade);
    int** M6 = alocaMatriz(metade);
    int** M7 = alocaMatriz(metade);

    int** Atemp = alocaMatriz(metade);
    int** Btemp = alocaMatriz(metade);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            soma(A11, A22, Atemp, metade);
            soma(B11, B22, Btemp, metade);
            strassen(Atemp, Btemp, M1, metade);
        }
    
        #pragma omp section
        {
            soma(A21, A22, Atemp, metade);
            strassen(Atemp, B11, M2, metade);
        }
    
        #pragma omp section
        {
            subtrai(B12, B22, Btemp, metade);
            strassen(A11, Btemp, M3, metade);
        }
    
        #pragma omp section
        {
            subtrai(B21, B11, Btemp, metade);
            strassen(A22, Btemp, M4, metade);
        }
    
        #pragma omp section
        {
            soma(A11, A12, Atemp, metade);
            strassen(Atemp, B22, M5, metade);
        }
    
        #pragma omp section
        {
            subtrai(A21, A11, Atemp, metade);
            soma(B11, B12, Btemp, metade);
            strassen(Atemp, Btemp, M6, metade);
        }
    
        #pragma omp section
        {
            subtrai(A12, A22, Atemp, metade);
            soma(B21, B22, Btemp, metade);
            strassen(Atemp, Btemp, M7, metade);
        }
    }

    soma(M1, M4, Atemp, metade);
    subtrai(Atemp, M5, Btemp, metade);
    soma(Btemp, M7, Atemp, metade);

    int** C12 = alocaMatriz(metade);
    int** C21 = alocaMatriz(metade);
    int** C22 = alocaMatriz(metade);

    soma(M3, M5, C12, metade);
    soma(M2, M4, C21, metade);
    subtrai(M1, M2, Btemp, metade);
    soma(Btemp, M3, Atemp, metade);
    soma(Atemp, M6, C22, metade);

    for (int i = 0; i < metade; i++) {
        for (int j = 0; j < metade; j++) {
            C[i][j] = Atemp[i][j];
            C[i][j + metade] = C12[i][j];
            C[i + metade][j] = C21[i][j];
            C[i + metade][j + metade] = C22[i][j];
        }
    }

    liberaMatriz(A11, metade); 
    liberaMatriz(A12, metade);
    liberaMatriz(A21, metade); 
    liberaMatriz(A22, metade);
    liberaMatriz(B11, metade); 
    liberaMatriz(B12, metade);
    liberaMatriz(B21, metade); 
    liberaMatriz(B22, metade);
    liberaMatriz(M1, metade); 
    liberaMatriz(M2, metade);
    liberaMatriz(M3, metade); 
    liberaMatriz(M4, metade);
    liberaMatriz(M5, metade); 
    liberaMatriz(M6, metade);
    liberaMatriz(M7, metade);
    liberaMatriz(Atemp, metade); 
    liberaMatriz(Btemp, metade);
    liberaMatriz(C12, metade); 
    liberaMatriz(C21, metade);
    liberaMatriz(C22, metade);
}

int binary_search(int valor, int *B, int end){
    int start = 0;

    while(start <= end){
        int mid = start + (end - start)/2;

        if(B[mid] == valor)
            return mid;
        
        if(valor > B[mid])
            start = mid + 1;
        else
            end = mid - 1;
    }

    return -1;
}

void P_Merge(int *C, int *A, int *B, int na, int nb){
    if(na < nb){
        P_Merge(C, B, A, nb, na);
        return;
    } else if(na == 0) {
        return;
    } else {
        int ma = na/2;
        int mb = binary_search(A[ma], B, nb);
        C[ma+mb] = A[ma];
        
        #pragma omp task
        P_Merge(C, A, B, ma, mb);

        #pragma omp task
        P_Merge(C + ma + mb, A + ma + 1, B + mb, na - ma - 1, nb - mb);

        #pragma omp taskwait
    }
}

void merge_sort(int *A, int *B, int size){
    if(size == 1){
        B[0] = A[0];
    } else {
        int *C = (int *) malloc(sizeof(int) * size);

        #pragma omp single
        {
            #pragma omp task
            merge_sort(C, A, size/2);

            #pragma omp task
            merge_sort(C + size/2, A + size/2, size - size/2);

            #pragma omp taskwait
        }

        P_Merge(B, C, C + size/2, size/2, size - size/2);
        free(C);
    }
}

int main(){
    srand(time(NULL));

    int **A = alocaMatriz(N);
    int **B = alocaMatriz(N);
    int **C = alocaMatriz(N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    printf("Matriz A:\n");
    imprime_matriz(A, N);
    printf("Matriz B:\n");
    imprime_matriz(B, N);

    printf("Multiplicação com Strassen:\n");
    strassen(A, B, C, N);
    imprime_matriz(C, N);

    int **C2 = alocaMatriz(N);
    #pragma omp parallel
    {
        #pragma omp single
        multi_matrizes(C2, 0, 0, A, 0, 0, B, 0, 0, N);
    }

    printf("Multiplicação paralela:\n");
    imprime_matriz(C2, N);

    int vetor[V];
    for (int i = 0; i < V; i++)
        vetor[i] = rand() % 100;

    int ordenado[V];
    printf("Vetor original:\n");
    imprime_vetor(vetor, V);

    merge_sort(vetor, ordenado, V);

    printf("Vetor ordenado (merge_sort paralelo):\n");
    imprime_vetor(ordenado, V);

    liberaMatriz(A, N);
    liberaMatriz(B, N);
    liberaMatriz(C, N);
    liberaMatriz(C2, N);

    return 0;
}
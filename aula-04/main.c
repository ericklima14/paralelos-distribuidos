#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>

int *soma_de_prefixos(int* vetor, int size){
    int *vetor_b = (int*)malloc(sizeof(int) * size * 2);
    int *vetor_p = (int*)malloc(sizeof(int) * size * 2);
    int logSize = (int) log2(size);

    #pragma omp parallel shared(vetor, size, logSize)
    {
        #pragma omp for
        for(int i = 0; i < size; i++){
            vetor_b[i + size] = vetor[i];
        }   

        for(int j = logSize; j >= 0; j--){
            int exp = (int)exp2(j);
            int exp_1 = (int)exp2(j + 1);
         
            #pragma omp for
            for(int i = exp; i < exp_1 - 1; i++){
                vetor_b[i] = vetor_b[2 * i] - vetor_b[2 * i - 1];
            }
        }

        vetor_p[1] = vetor_b[1];

        for(int j = 1; j < logSize; j++){
            int exp = (int)exp2(j);
            int exp_1 = (int)exp2(j + 1);
         
            #pragma omp for
            for(int i = exp; i < exp_1 - 1; i++){
                if(i % 2 == 0){
                    vetor_p[i] = vetor_p[(int)i/2] - vetor_b[i + 1];
                } else {
                    vetor_p[i] = vetor_p[(int)(i - 1)/2];
                }
            }
        }
    }

    return vetor_p;
}



int main(){


    return 0;
}
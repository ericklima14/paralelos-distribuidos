#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int exercicio1(){
    int maior = 0;
    int i = 0;
    int vetor[] = {1, 2, 5, 4, 2};

    #pragma omp parallel shared(maior, vetor)
    {
        #pragma omp for
        for(i = 0; i < 5; i++){
            //TODO: arrumar essa funcao para parar de ser sincrona
            if(maior < vetor[i]){
                maior = vetor[i];
            }
        }
    }

    return maior;
}

int* exercicio2(int *vetor){
    int vetor1[] = {1 , 2, 3};
    int vetor2[] = {1 , 2, 3};
    int i = 0;

    #pragma omp parallel shared(vetor, i)
    {
        #pragma omp for
        for(i = 0; i < 3; i++){
            vetor[i] = vetor1[i] + vetor2[i];
        }
    }
    
    return vetor;
}

int exercicio3(){
    int v[] = {1, 2, 3};
    int min = 0;

    #pragma omp parallel shared(v, min)
    {
        int m[3][3];
        int x[3] = {0, 0, 0};

        #pragma omp for
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                if(v[i] > v[j]){
                    m[i][j] = 1;
                } else {
                    m[i][j] = 0;
                }
            }
        }

        #pragma omp for
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                if(m[i][j] == 1){
                   x[i] = 1;
                }
            }
        }

        #pragma omp for
        for(int i = 0; i < 3; i++){
            if(x[i] == 0){
                min = v[i];
            }
        }

    }

    return min;
}

int main(int argc, char* argv[]) {
    int *vetor = (int *)malloc(3 * sizeof(int));
    vetor = exercicio2(vetor);

    printf("Maior valor do vetor é: %d\n", exercicio1());

    for(int i = 0; i < 3; i++){
        printf("O valor do elemento %d é: %d\n", i, vetor[i]);
    }

    printf("Menor valor do vetor é: %d\n", exercicio3());

    free(vetor);

    return 0;
}
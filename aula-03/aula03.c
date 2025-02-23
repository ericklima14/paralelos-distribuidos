#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>

int primeiroAlgoritmo(int *vetor, int size){
    int logSize = (int)log2(size);

    #pragma omp parallel shared(vetor, size, logSize)
    {
        for(int k = 0; k < logSize; k++){
            int expK = (int)exp2(k);
            #pragma omp for
            for(int i = 0; i < size; i = i + expK){
                if(i + expK < size){
                    vetor[i] = (int)fmax(vetor[i], vetor[i + expK]);
                }
            }
        }
    }

    return vetor[0];
}

int segundoAlgoritmo(int *vetor, int size){
    int num_proc = omp_get_max_threads();

    #pragma omp parallel shared(vetor, size, num_proc)
    {
        #pragma omp for
        for(int i = 0; i < num_proc; i++){
            int inicio = i * (size/num_proc);
            for(int k = inicio + 1; k < inicio + (size/num_proc); k++){
                if(vetor[k] > vetor[inicio]){
                    vetor[inicio] = vetor[k];
                }
            }
        }

        int logSize = (int)log2(size);
        int logNP = (int)log2(size/num_proc);

        for(int k = logNP + 1; k < logSize; k++){
            int expK = (int)exp2(k);
            #pragma omp for
            for(int i = 0; i < size; i = i + expK){
                if(i + expK < size){
                    vetor[i] = fmax(vetor[i], vetor[i + expK]);
                }
            }
        }

    }
    
    return vetor[0];
}

int terceiroAlgoritmo(int *vetor, int size){
    int *vetor_resultante = (int *)malloc(sizeof(int) * (size * 2));

    #pragma omp parallel shared(vetor, size, vetor_resultante)
    {
        #pragma omp for
        for(int i = 0; i < size; i++){
            vetor_resultante[size + i] = vetor[i];
        }

        int log_n = (int)log2(size);

        for(int j = log_n - 1; j >= 0; j--){
            int exp_j = (int) exp2(j);
            int exp_maior = (int) exp2(j + 1);

            #pragma omp for
            for(int i = exp_j; i < exp_maior; i++){
                vetor_resultante[i] = (int)fmax(vetor_resultante[2 * i], vetor_resultante[2*i + 1]);
            }
        }
    }

    int resultado = vetor_resultante[1];
    free(vetor_resultante);

    return resultado; 
}

int quartoAlgoritmo(int *vetor, int size){
    int num_proc = omp_get_max_threads();
    int *vetor_res = (int *) malloc(sizeof(int) * (num_proc * 2));

    #pragma omp parallel shared(vetor, size, vetor_res, num_proc)
    {
        #pragma omp for
        for(int i = 0; i < num_proc; i++){
            vetor_res[num_proc + i] = vetor[i * (size/num_proc)];
            for(int j = 1; j < (size/num_proc); j++){
                if(vetor[i * (size/num_proc) + j] > vetor_res[num_proc + i]){
                    vetor_res[num_proc + i] = vetor[i * (size/num_proc) + j] ;
                }
            }
        }

        int log_p = (int) log2(num_proc);

        for(int j = log_p - 1; j >= 0; j--){
            int exp_j = (int)exp2(j);
            int exp_maior = (int)exp2(j + 1);
            
            #pragma omp for
            for(int i = exp_j; i < exp_maior; i++){
                vetor_res[i] = (int)fmax(vetor_res[2 * i], vetor_res[2* i + 1]);
            }
        }
    }
    
    int resultado = vetor_res[1];
    free(vetor_res);

    return resultado;
}

int main(){
    int vetor[8] = {15, 39, 27, 51, 32, 67, 10, 41};
    int vetor_maior[32];

    srand(time(NULL));

    printf("Vetor Maior -- Random\n\n[");
    for(int i = 0; i < 32; i++){
        vetor_maior[i] = rand() % 100 + 1;
        printf("%d, ", vetor_maior[i]);
    }
    printf("]\n\n");

    printf("O resultado do primeiro algoritmo é: %d\n", primeiroAlgoritmo(vetor, 8));
    printf("O resultado do segundo algoritmo é: %d\n", segundoAlgoritmo(vetor, 8));
    printf("O resultado do terceiro algoritmo é: %d\n", terceiroAlgoritmo(vetor, 8));
    printf("O resultado do quarto algoritmo é: %d\n", quartoAlgoritmo(vetor, 8));

    printf("\n-----------------------------------------------\n\n");
    printf("O maior resultado do vetor para o segundo algoritmo é: %d\n", segundoAlgoritmo(vetor_maior, 32));
    printf("O maior resultado do vetor para o quarto algoritmo é: %d\n", quartoAlgoritmo(vetor_maior, 32));

    return 0;
}
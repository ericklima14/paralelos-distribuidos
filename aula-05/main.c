#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>


void merge(int* vetor, int first_index, int meio, int size){
    int i, j, k;
    int n1 = meio - first_index + 1;
    int n2 = size - meio;
    
    int array_left[n1], array_right[n2];

    for(i = 0; i < n1; i++){
        array_left[i] = vetor[first_index + i];
    }

    for(j = 0; j < n2; j++){
        array_right[j] = vetor[meio + 1 + j];
    }

    i = 0;
    j = 0;
    k = first_index;

    while(i < n1 && j < n2){
        if(array_left[i] <= array_right[j]){
            vetor[k] = array_left[i];
            i++;
        } else {
            vetor[k] = array_right[j];
            j++;
        }

        k++;
    }

    while(i < n1){
        vetor[k] = array_left[i];
        i++;
        k++;
    }

    while(j < n2){
        vetor[k] = array_right[j];
        j++;
        k++;
    }
}

void merge_sort_paralelo(int * vetor, int first_index, int size){
    if(first_index < size){
        int meio = first_index + (size - first_index)/2;
        int meio_1 = meio + 1;

        #pragma omp parallel sections
        {
            #pragma omp section
            merge_sort_paralelo(vetor, first_index, meio);

            #pragma omp section
            merge_sort_paralelo(vetor, meio_1, size);
        }

        merge(vetor, first_index, meio, size);
    }
}


void printArray(int A[], int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}

void merge_sort(int * vetor, int first_index, int size){
    if(first_index < size){
        int meio = first_index + (size - first_index)/2;
        int meio_1 = meio + 1;

        merge_sort(vetor, first_index, meio);
        merge_sort(vetor, meio_1, size);
        merge(vetor, first_index, meio, size);
    }
}

void merge_sort_iterativo(int* vetor, int size) {
    for (int i = 1; i < size; i *= 2) {
        for (int j = 0; j < size - 1; j += 2 * i) {
            int meio = j + i - 1;
            int right_end = (j + 2 * i - 1 < size - 1) ? j + 2 * i - 1 : size - 1;

            if (meio < right_end) { 
                merge(vetor, j, meio, right_end);
            }
        }
    }
}

void merge_sort_iterativo_paralelo(int* vetor, int size) {
    for (int i = 1; i < size; i *= 2) {
        #pragma omp parallel for
        for (int j = 0; j < size - 1; j += 2 * i) {
            int meio = j + i - 1;
            int right_end = (j + 2 * i - 1 < size - 1) ? j + 2 * i - 1 : size - 1;

            if (meio < right_end) { 
                merge(vetor, j, meio, right_end);
            }
        }
    }
}

int main(){
    int size = 100000;
    int vetor_a[size];
    int vetor_b[size];
    int vetor_c[size];
    int vetor_d[size];
    srand(time(NULL));

    int thread_counts[] = {2, 4, 8, 16}; // Testando com 2, 4, 8 e 16 threads

    for(int t = 0; t < 4; t++){
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);

        double tempo_medio_seq = 0, tempo_medio_par = 0;
        double tempo_medio_seq_ite = 0, tempo_medio_par_ite = 0;

        for(int j = 0; j < 10; j++){
            for(int i = 0; i < size; i++){
                int random = rand() % size + 1;
                vetor_a[i] = random;
                vetor_b[i] = random;
                vetor_c[i] = random;
                vetor_d[i] = random;
            }

            clock_t start_seq = clock();
            merge_sort(vetor_a, 0, size - 1);
            clock_t end_seq = clock();
            tempo_medio_seq += (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

            clock_t start_par = clock();
            merge_sort_paralelo(vetor_b, 0, size - 1);
            clock_t end_par = clock();
            tempo_medio_par += (double)(end_par - start_par) / CLOCKS_PER_SEC;

            start_seq = clock();
            merge_sort_iterativo(vetor_c, size);
            end_seq = clock();
            tempo_medio_seq_ite += (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

            start_par = clock();
            merge_sort_iterativo_paralelo(vetor_d, size);
            end_par = clock();
            tempo_medio_par_ite += (double)(end_par - start_par) / CLOCKS_PER_SEC;
        }

        printf("-------------------------------------------------------------\n");
        printf("TEMPO COM %d THREADS:\n", num_threads);
        printf("TEMPO MÉDIO SEQUENCIAL: %.6f segundos\n", tempo_medio_seq / 10);
        printf("TEMPO MÉDIO PARALELO  : %.6f segundos\n", tempo_medio_par / 10);
        printf("TEMPO MÉDIO SEQUENCIAL ITERATIVO: %.6f segundos\n", tempo_medio_seq_ite / 10);
        printf("TEMPO MÉDIO PARALELO ITERATIVO : %.6f segundos\n", tempo_medio_par_ite / 10);
    }

    return 0;
}

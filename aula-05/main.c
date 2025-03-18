#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

void merge_paralelo(int* vetor, int first_index, int meio, int size){
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

        merge_paralelo(vetor, first_index, meio, size);
    }
}


void printArray(int A[], int size)
{
    int i;
    for (i = 0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}

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

void merge_sort(int * vetor, int first_index, int size){
    if(first_index < size){
        int meio = first_index + (size - first_index)/2;
        int meio_1 = meio + 1;

        merge_sort(vetor, first_index, meio);
        merge_sort(vetor, meio_1, size);
        merge(vetor, first_index, meio, size);
    }
}

int main(){
    int size = 10;
    int vetor_a[10];
    int vetor_b[10];
    srand(time(NULL));

    for(int i = 0; i < size; i++){
        int random = rand() % size + 1;
        vetor_a[i] = random;
        vetor_b[i] = random;
    }

    printArray(vetor_a, size);
    printArray(vetor_b, size);

    double st1 = omp_get_wtime();
    merge_sort(vetor_a, 0, size - 1);
    double en1 = omp_get_wtime();

    double st2 = omp_get_wtime();
    merge_sort_paralelo(vetor_b, 0, size - 1);
    double en2 = omp_get_wtime();

    printf("-------------------------------------------------------------\n");
    printArray(vetor_a, size);
    printArray(vetor_b, size);

    printf("TEMPO FINAL SEQUENCIAL: %lf\n", en1 - st1);
    printf("TEMPO FINAL PARALERO: %lf\n", en2 - st2);
    return 0;
}

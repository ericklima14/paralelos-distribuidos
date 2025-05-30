// mergesort_search.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// mescla arr[l..m] e arr[m+1..r] em tmp[]
void merge(int *arr, int l, int m, int r, int *tmp) {
    int i = l, j = m+1, k = 0;
    while (i <= m && j <= r)
        tmp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    while (i <= m) tmp[k++] = arr[i++];
    while (j <= r) tmp[k++] = arr[j++];
    for (i = 0; i < k; i++)
        arr[l + i] = tmp[i];
}

// recursão do MergeSort
void mergesort_rec(int *arr, int l, int r, int *tmp) {
    if (l < r) {
        int m = (l + r) / 2;
        #pragma omp task shared(arr,tmp) if(r-l > 1000)
        mergesort_rec(arr, l, m, tmp);
        #pragma omp task shared(arr,tmp) if(r-l > 1000)
        mergesort_rec(arr, m+1, r, tmp);
        #pragma omp taskwait
        merge(arr, l, m, r, tmp);
    }
}

void mergesort(int *arr, int n) {
    int *tmp = malloc(n * sizeof(int));
    #pragma omp parallel
    {
        #pragma omp single
        mergesort_rec(arr, 0, n-1, tmp);
    }
    free(tmp);
}

// busca binária iterativa em arr[0..n-1]
int binary_search(int *arr, int n, int key) {
    int l = 0, r = n-1;
    while (l <= r) {
        int m = l + (r - l)/2;
        if (arr[m] == key)      return m;
        else if (arr[m] < key)  l = m + 1;
        else                    r = m - 1;
    }
    return -1;  // não encontrado
}

int main() {
    int n = 1000000;
    int *arr = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        arr[i] = rand();

    double t0 = omp_get_wtime();
    mergesort(arr, n);
    double t1 = omp_get_wtime();
    printf("MergeSort: %f s\n", t1 - t0);

    int key = arr[n/2];
    int idx = binary_search(arr, n, key);
    printf("BuscaBinária: chave %d encontrada no índice %d\n", key, idx);

    free(arr);
    return 0;
}
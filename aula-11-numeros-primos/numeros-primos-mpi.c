#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int eh_primo(long int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    long int i;
    for (i = 3; i <= sqrt(n); i += 2)
        if (n % i == 0)
            return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    int id, qtd, total_local = 0, total_global = 0;
    long int n = 7000000; 

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &qtd);

    long int inicio, fim, i;

    long int bloco = (n - 2) / qtd;  
    inicio = 3 + id * bloco;
    fim = (id == qtd - 1) ? n : inicio + bloco - 1;

    if (inicio % 2 == 0) inicio++; 

    for (i = inicio; i <= fim; i += 2)
        if (eh_primo(i))
            total_local++;

    if (id == 0)
        total_local += 1; 

    MPI_Reduce(&total_local, &total_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (id == 0)
        printf("Quant. de primos entre 1 e %ld: %d\n", n, total_global);

    MPI_Finalize();
    return 0;
}
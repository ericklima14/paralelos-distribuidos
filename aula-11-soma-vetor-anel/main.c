#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define TAM 8

int main(int argc, char** argv) {
    int id, qtd, anterior, proximo;
    int *vetor = NULL;
    int tam_local, soma_local = 0, soma_total;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &qtd);

    tam_local = TAM / qtd;

    int* parte = malloc(tam_local * sizeof(int));

    if (id == 0) {
        vetor = malloc(TAM * sizeof(int));
        for (int i = 0; i < TAM; i++)
            vetor[i] = i + 1;  
    }

    MPI_Scatter(vetor, tam_local, MPI_INT, parte, tam_local, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < tam_local; i++)
        soma_local += parte[i];

    anterior = (id - 1 + qtd) % qtd;
    proximo = (id + 1) % qtd;

    if (id == 0) {
        MPI_Send(&soma_local, 1, MPI_INT, proximo, 0, MPI_COMM_WORLD);
        MPI_Recv(&soma_total, 1, MPI_INT, anterior, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Soma total: %d\n", soma_total);
    } else {
        int acumulado;
        MPI_Recv(&acumulado, 1, MPI_INT, anterior, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        acumulado += soma_local;
        MPI_Send(&acumulado, 1, MPI_INT, proximo, 0, MPI_COMM_WORLD);
    }

    free(parte);
    if (vetor) free(vetor);

    MPI_Finalize();
    return 0;
}
    
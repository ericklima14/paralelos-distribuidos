#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define TAM 4

void mostrar(int *m, int linhas, int colunas) {
    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < colunas; j++)
            printf("%4d ", m[i * colunas + j]);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int id, qtd;
    int *a, *b, *c, *parte_a, *parte_c;
    int fatia;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &qtd);

    fatia = TAM / qtd;

    parte_a = malloc(fatia * TAM * sizeof(int));
    parte_c = malloc(fatia * TAM * sizeof(int));
    b = malloc(TAM * TAM * sizeof(int));

    if (id == 0) {
        a = malloc(TAM * TAM * sizeof(int));
        c = malloc(TAM * TAM * sizeof(int));

        for (int i = 0; i < TAM; i++)
            for (int j = 0; j < TAM; j++) {
                a[i * TAM + j] = i + j;
                b[i * TAM + j] = i == j ? 1 : 0;
            }
    }

    MPI_Bcast(b, TAM * TAM, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(a, fatia * TAM, MPI_INT, parte_a, fatia * TAM, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < fatia; i++) {
        for (int j = 0; j < TAM; j++) {
            parte_c[i * TAM + j] = 0;
            for (int k = 0; k < TAM; k++)
                parte_c[i * TAM + j] += parte_a[i * TAM + k] * b[k * TAM + j];
        }
    }

    MPI_Gather(parte_c, fatia * TAM, MPI_INT, c, fatia * TAM, MPI_INT, 0, MPI_COMM_WORLD);

    if (id == 0) {
        printf("Resultado:\n");
        mostrar(c, TAM, TAM);
        free(a);
        free(c);
    }

    free(parte_a);
    free(parte_c);
    free(b);

    MPI_Finalize();
    return 0;
}
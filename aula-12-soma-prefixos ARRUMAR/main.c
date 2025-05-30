#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2) {
        if (rank == 0) fprintf(stderr, "Uso: %s <tamanho_total_n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    if (n % p != 0) {
        if (rank == 0) fprintf(stderr,
            "Erro: n (%d) deve ser múltiplo de p (%d)\n", n, p);
        MPI_Finalize();
        return 1;
    }

    int r = n / p;
    double *B = malloc(r * sizeof(double));
    // --- Inicializa localmente o subvetor B ---
    // Aqui: B[k] = rank*r + k + 1 apenas para teste (1,2,3,...)
    for (int k = 0; k < r; k++)
        B[k] = rank * r + k + 1;

    // --- Passo 1: soma local si ---
    double local_sum = 0.0;
    for (int k = 0; k < r; k++)
        local_sum += B[k];

    // --- Passo 2: troca de somas (uma rodada de comunicação) ---
    // Todos reúnem todos os local_sum em all_sums[0..p-1]
    double *all_sums = malloc(p * sizeof(double));
    MPI_Allgather(
        &local_sum, 1, MPI_DOUBLE,
        all_sums,    1, MPI_DOUBLE,
        MPI_COMM_WORLD
    );  // :contentReference[oaicite:1]{index=1}

    // --- Passo 3: cada Pi computa S[i*r] = sum_{j=0..i} all_sums[j] ---
    double prefix_up_to_block = 0.0;
    for (int i = 0; i <= rank; i++)
        prefix_up_to_block += all_sums[i];

    // --- Passo 4 e 5: computa localmente os prefixos de B ---
    double *S_local = malloc(r * sizeof(double));
    // primeiro elemento do bloco:
    S_local[0] = prefix_up_to_block - local_sum + B[0];
    // prefixo dentro do bloco
    for (int k = 1; k < r; k++)
        S_local[k] = S_local[k-1] + B[k];

    // --- (Opcional) Coleta e impressão no processo 0 ---
    double *S = NULL;
    if (rank == 0) S = malloc(n * sizeof(double));
    MPI_Gather(
        S_local, r, MPI_DOUBLE,
        S,       r, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("Prefix sums de 1..%d:\n", n);
        for (int i = 0; i < n; i++)
            printf("%.0f ", S[i]);
        printf("\n");
        free(S);
    }

    free(B);
    free(all_sums);
    free(S_local);
    MPI_Finalize();
    return 0;
}
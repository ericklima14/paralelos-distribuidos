#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int cmp_double(const void *a, const void *b) {
    double x = *(double*)a, y = *(double*)b;
    return (x < y) ? -1 : (x > y);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 2) {
        if (rank == 0)
            fprintf(stderr, "Uso: %s <tamanho_total_n>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int n = atoi(argv[1]);
    if (n % p != 0) {
        if (rank == 0)
            fprintf(stderr, "Erro: n (%d) deve ser múltiplo de p (%d)\n", n, p);
        MPI_Finalize();
        return 1;
    }

    int r = n / p;
    double *local = malloc(r * sizeof(double));

    for (int i = 0; i < r; i++)
        local[i] = rank * r + i + 1;

    qsort(local, r, sizeof(double), cmp_double);

    double *samples = malloc((p-1) * sizeof(double));
    for (int i = 1; i < p; i++) {
        int idx = i * r / p;
        samples[i-1] = local[idx];
    }

    double *all_samples = malloc(p * (p-1) * sizeof(double));
    MPI_Allgather(
        samples, p-1, MPI_DOUBLE,
        all_samples, p-1, MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    qsort(all_samples, p*(p-1), sizeof(double), cmp_double);
    double *splitters = malloc((p-1) * sizeof(double));
    for (int i = 1; i < p; i++) {
        splitters[i-1] = all_samples[i * p];
    }

    int *send_counts = calloc(p, sizeof(int));
    for (int i = 0; i < r; i++) {
        double v = local[i];
        int b = 0;
        while (b < p-1 && v >= splitters[b]) b++;
        send_counts[b]++;
    }

    int *recv_counts = malloc(p * sizeof(int));
    MPI_Alltoall(
        send_counts, 1, MPI_INT,
        recv_counts, 1, MPI_INT,
        MPI_COMM_WORLD
    );

    int *send_disp = malloc(p * sizeof(int));
    int *recv_disp = malloc(p * sizeof(int));
    send_disp[0] = recv_disp[0] = 0;
    for (int i = 1; i < p; i++) {
        send_disp[i] = send_disp[i-1] + send_counts[i-1];
        recv_disp[i] = recv_disp[i-1] + recv_counts[i-1];
    }
    int total_recv = recv_disp[p-1] + recv_counts[p-1];

    double *send_buf = malloc(r * sizeof(double));
    int *pos = calloc(p, sizeof(int));
    for (int i = 0; i < r; i++) {
        double v = local[i];
        int b = 0;
        while (b < p-1 && v >= splitters[b]) b++;
        send_buf[ send_disp[b] + pos[b]++ ] = v;
    }

    double *recv_buf = malloc(total_recv * sizeof(double));
    MPI_Alltoallv(
        send_buf, send_counts, send_disp, MPI_DOUBLE,
        recv_buf, recv_counts, recv_disp, MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    qsort(recv_buf, total_recv, sizeof(double), cmp_double);

    double *sorted = NULL;
    int *gather_counts = NULL, *gather_disp = NULL;
    if (rank == 0) {
        sorted        = malloc(n * sizeof(double));
        gather_counts = malloc(p * sizeof(int));
    }
    MPI_Gather(
        &total_recv, 1, MPI_INT,
        gather_counts, 1, MPI_INT,
        0, MPI_COMM_WORLD
    );
    if (rank == 0) {
        gather_disp = malloc(p * sizeof(int));
        gather_disp[0] = 0;
        for (int i = 1; i < p; i++)
            gather_disp[i] = gather_disp[i-1] + gather_counts[i-1];
    }
    MPI_Gatherv(
        recv_buf, total_recv, MPI_DOUBLE,
        sorted,  gather_counts, gather_disp, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("Vetor totalmente ordenado:\n");
        for (int i = 0; i < n; i++)
            printf("%.0f ", sorted[i]);
        printf("\n");
        free(sorted);
        free(gather_counts);
        free(gather_disp);
    }

    free(local);
    free(samples);
    free(all_samples);
    free(splitters);
    free(send_counts);
    free(recv_counts);
    free(send_disp);
    free(recv_disp);
    free(send_buf);
    free(recv_buf);
    free(pos);

    MPI_Finalize();
    return 0;
}
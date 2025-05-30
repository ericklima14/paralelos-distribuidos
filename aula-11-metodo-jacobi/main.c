#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define CONV_THRESHOLD 1.0e-5f

double **grid;
double **new_grid;
int size;
int iter_max_num;

double absolute(double num) {
    if (num < 0)
        return -1.0 * num;
    return num;
}

void alocar_memoria(int linhas) {
    grid = malloc(linhas * sizeof(double *));
    new_grid = malloc(linhas * sizeof(double *));
    for (int i = 0; i < linhas; i++) {
        grid[i] = malloc(size * sizeof(double));
        new_grid[i] = malloc(size * sizeof(double));
    }
}

void inicializar_grid(int linhas, int desloc) {
    int centro = size / 2;
    int linf = centro - (size / 10);
    int lsup = centro + (size / 10);
    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < size; j++) {
            int global_i = i + desloc;
            if (global_i >= linf && global_i <= lsup && j >= linf && j <= lsup)
                grid[i][j] = 100;
            else
                grid[i][j] = 0;
            new_grid[i][j] = 0.0;
        }
    }
}

void salvar_grid(double **final, int linhas, int rank) {
    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < size; j++) {
            printf("%lf ", final[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int id, qtd, iter = 0, erro, globalError;
    size = 2048;
    iter_max_num = 15000;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &qtd);

    int linhas = size / qtd;
    int inicio = id * linhas;
    int fim = (id == qtd - 1) ? size : inicio + linhas;
    int linhas_local = fim - inicio + 2;

    alocar_memoria(linhas_local);
    inicializar_grid(linhas_local, inicio - 1);

    do {
        erro = 0;

        if (id > 0)
            MPI_Send(grid[1], size, MPI_DOUBLE, id - 1, 0, MPI_COMM_WORLD);
        if (id < qtd - 1)
            MPI_Recv(grid[linhas_local - 1], size, MPI_DOUBLE, id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (id < qtd - 1)
            MPI_Send(grid[linhas_local - 2], size, MPI_DOUBLE, id + 1, 1, MPI_COMM_WORLD);
        if (id > 0)
            MPI_Recv(grid[0], size, MPI_DOUBLE, id - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i < linhas_local - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                new_grid[i][j] = 0.25 * (grid[i][j + 1] + grid[i][j - 1] +
                                         grid[i - 1][j] + grid[i + 1][j]);
                if (absolute(new_grid[i][j] - grid[i][j]) > CONV_THRESHOLD)
                    erro = 1;
            }
        }

        for (int i = 1; i < linhas_local - 1; i++) {
            for (int j = 1; j < size - 1; j++) {
                grid[i][j] = new_grid[i][j];
            }
        }

        MPI_Allreduce(&erro, &globalError, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        iter++;
    } while (globalError && iter < iter_max_num);

    if (id == 0) {
        double **final_grid = malloc(size * sizeof(double *));
        for (int i = 0; i < size; i++)
            final_grid[i] = malloc(size * sizeof(double));
        for (int i = 0; i < linhas; i++)
            for (int j = 0; j < size; j++)
                final_grid[i][j] = grid[i + 1][j];
        for (int p = 1; p < qtd; p++) {
            for (int i = 0; i < linhas; i++)
                MPI_Recv(final_grid[p * linhas + i], size, MPI_DOUBLE, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        salvar_grid(final_grid, size, id);
        for (int i = 0; i < size; i++)
            free(final_grid[i]);
        free(final_grid);
    } else {
        for (int i = 0; i < linhas; i++)
            MPI_Send(grid[i + 1], size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    for (int i = 0; i < linhas_local; i++) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);

    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void sum_parallel_mpi(char* output, unsigned long d, unsigned long n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long* local_digits = (unsigned long*)calloc(d + 11, sizeof(unsigned long));
    unsigned long* global_digits = (unsigned long*)calloc(d + 11, sizeof(unsigned long));

    for (unsigned long i = rank + 1; i <= n; i += size) {
        unsigned long remainder = 1;
        for (unsigned long digit = 0; digit < d + 11 && remainder; ++digit) {
            unsigned long div = remainder / i;
            unsigned long mod = remainder % i;
            local_digits[digit] += div;
            remainder = mod * 10;
        }
    }

    MPI_Reduce(local_digits, global_digits, d + 11, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (unsigned long i = d + 11 - 1; i > 0; --i) {
            global_digits[i - 1] += global_digits[i] / 10;
            global_digits[i] %= 10;
        }

        if (global_digits[d + 1] >= 5) {
            ++global_digits[d];
        }

        for (unsigned long i = d; i > 0; --i) {
            global_digits[i - 1] += global_digits[i] / 10;
            global_digits[i] %= 10;
        }

        int len = sprintf(output, "%lu.", global_digits[0]);
        for (unsigned long i = 1; i <= d; ++i) {
            len += sprintf(output + len, "%lu", global_digits[i]);
        }
        output[len] = '\0';
    }

    free(local_digits);
    free(global_digits);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    char output[1000];
    unsigned long d = 12; 
    unsigned long n = 7;  

    sum_parallel_mpi(output, d, n);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        printf("Resultado MPI: %s\n", output);
    }

    MPI_Finalize();
    return 0;
}
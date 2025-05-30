#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void sum_parallel_omp(char* output, const unsigned long d, const unsigned long n) {
    unsigned long* digits = (unsigned long*)calloc(d + 11, sizeof(unsigned long));

    #pragma omp parallel
    {
        unsigned long* local_digits = (unsigned long*)calloc(d + 11, sizeof(unsigned long));

        #pragma omp for
        for (unsigned long i = 1; i <= n; ++i) {
            unsigned long remainder = 1;
            for (unsigned long digit = 0; digit < d + 11 && remainder; ++digit) {
                unsigned long div = remainder / i;
                unsigned long mod = remainder % i;
                local_digits[digit] += div;
                remainder = mod * 10;
            }
        }

        #pragma omp critical
        for (unsigned long i = 0; i < d + 11; ++i) {
            digits[i] += local_digits[i];
        }

        free(local_digits);
    }

    for (unsigned long i = d + 11 - 1; i > 0; --i) {
        digits[i - 1] += digits[i] / 10;
        digits[i] %= 10;
    }

    if (digits[d + 1] >= 5) {
        ++digits[d];
    }

    for (unsigned long i = d; i > 0; --i) {
        digits[i - 1] += digits[i] / 10;
        digits[i] %= 10;
    }

    int len = sprintf(output, "%lu.", digits[0]);
    for (unsigned long i = 1; i <= d; ++i) {
        len += sprintf(output + len, "%lu", digits[i]);
    }
    output[len] = '\\0';

    free(digits);
}

int main() {
    char output[1000];
    unsigned long d = 12;  
    unsigned long n = 7; 

    sum_parallel_omp(output, d, n);

    printf("Resultado OpenMP: %s\n", output);
    return 0;
}
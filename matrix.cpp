#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <xmmintrin.h>

#define SIZE 480
#define NUMTHREADS 24
#define BIL 1000000000.0

typedef struct {
  pthread_t thread;
  int i;
} thread_data;

float x[SIZE][SIZE];
float y[SIZE][SIZE];
float ret[SIZE][SIZE];
thread_data threads[NUMTHREADS];


void print_matrix(float mat[SIZE][SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      printf("%f ", mat[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void matrix_mult(float A[SIZE][SIZE], float B[SIZE][SIZE], float ans[SIZE][SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      for (int k = 0; k < SIZE; k++) {
        ans[i][j] += (A[i][k] * B[k][j]);
      }
    }
  }
}
    
void * matrix_mult_worker(void * args) {
  thread_data * td = (thread_data*) args;
  int div = SIZE / NUMTHREADS;
  int start = td->i * div;
  int end = start + div;
  for (int i = 0; i < SIZE; i++) {
    for (int j = start; j < end; j++) {
      for (int k = 0; k < SIZE; k++) {
        ret[i][j] += (x[k][j] * y[i][k]);
      }
    }
  }
  return NULL;
}

void matrix_transpose(float A[SIZE][SIZE], float ans[SIZE][SIZE]) {
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      ans[j][i] = A[i][j];
    }
  }
}

void matrix_mult_simd(float A[SIZE][SIZE], float B[SIZE][SIZE], float ans[SIZE][SIZE]) {
  float temp[4] = {0};
  __m128 acc, a, b;

  for (int i = 0 ; i < SIZE ; i++) {
    for (int j = 0; j < SIZE; j ++) {
      acc = _mm_set1_ps(0.0);
      for (int k = 0; k < (SIZE - 3); k +=4) {
        a = _mm_loadu_ps(&A[j][k]);
        b = _mm_loadu_ps(&B[j][k]);
        acc = _mm_add_ps(acc, _mm_mul_ps(a, b));
      }
      _mm_storeu_ps(temp, acc);
      ans[i][j] = temp[0] + temp[1] + temp[2] + temp[3];
    }
  }
}

int main() {
  float ans[SIZE][SIZE];
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      x[i][j] = j + 1;
      y[i][j] = j + 1;
      ret[i][j] = 0;
      ans[i][j] = 0;
    }
  }

  struct timespec start, end;
    
  clock_gettime(CLOCK_MONOTONIC, &start);
  matrix_mult(x, y, ans);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double ref_time = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / BIL;

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < NUMTHREADS; i++) {
    threads[i].i = i;
    pthread_create(&threads[i].thread, NULL, matrix_mult_worker, &threads[i]);
  }
  for (int i = 0; i < NUMTHREADS; i++) {
    pthread_join(threads[i].thread, NULL);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  double threaded = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / BIL;

  float transpose[SIZE][SIZE];
  matrix_transpose(y, transpose);
  clock_gettime(CLOCK_MONOTONIC, &start);
  matrix_mult_simd(x, y, ans);
  clock_gettime(CLOCK_MONOTONIC, &end);
  double simd = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / BIL;

  printf("Nonthreaded: %f\nThreaded: %f (%.2fx faster)\nSIMD: %f (%.2fx faster)\n", ref_time, threaded, ref_time/threaded, simd, ref_time/simd);
}

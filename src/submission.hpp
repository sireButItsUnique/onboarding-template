#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <immintrin.h>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  size_t rows_;
  size_t cols_;
  size_t size_;

  double* data_;

public:
  Grid(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;

    size_ = rows * cols * sizeof(double);
    data_ = (double*)malloc(size_);
    memset(data_, 0, size_);
  }

  ~Grid() {
    free(data_);
  }

  Grid& operator=(const Grid& other) {
    free(data_);
    rows_ = other.rows_;
    cols_ = other.cols_;
    size_ = other.size_;
    
    data_ = (double*)malloc(size_);
    memcpy(data_, other.data_, size_);
    return *this;
  }

  size_t get_rows() const {
    return rows_;
  }

  size_t get_cols() const {
    return cols_;
  }
  
  double* row_ptr(size_t i) {
    return data_ + i * cols_; 
  }
  
  const double* row_ptr(size_t i) const { 
    return data_ + i * cols_; 
  }

  double& operator()(size_t row, size_t col) {
    size_t i = row * cols_ + col; 
    return data_[i];
  }

  double  operator()(size_t row, size_t col) const {
    size_t i = row * cols_ + col;
    return data_[i];
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
__attribute__((target("avx2,fma")))
void apply_stencil_row(const double* __restrict__ c, const double* __restrict__ up, const double* __restrict__ down,
                       double* __restrict__ dst, const size_t m, const __m256d& half, const __m256d& eighth) {
  
  size_t j = 1;
  for (; j + 4 <= m - 1; j += 4) { // run 4 at a time for interior points only
    __m256d vcur = _mm256_loadu_pd(c + j);
    __m256d vup = _mm256_loadu_pd(up + j);
    __m256d vdown = _mm256_loadu_pd(down + j);
    __m256d vleft = _mm256_loadu_pd(c + j - 1);
    __m256d vright = _mm256_loadu_pd(c + j + 1);

    __m256d neighbors = _mm256_add_pd(_mm256_add_pd(vup, vdown), _mm256_add_pd(vleft, vright));
    __m256d res = _mm256_add_pd(_mm256_mul_pd(half, vcur), _mm256_mul_pd(eighth, neighbors));
    _mm256_storeu_pd(dst + j, res);
  }

  // scalar for leftover
  for (; j < m - 1; j++) {
    dst[j] = 0.5 * c[j] + 0.125 * (up[j] + down[j] + c[j - 1] + c[j + 1]);
  }
}

__attribute__((target("avx2,fma")))
void apply_stencil(const Grid& old, Grid& res) {
  size_t n = old.get_rows();
  size_t m = old.get_cols();

  // no interior points to update
  if (n < 3 || m < 3) {
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        res(i, j) = old(i, j);
      }
    }
    return;
  } 

  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d eighth = _mm256_set1_pd(0.125);
  
  /*
    threading seems to just make it slower, bottleneck might be moving memory so the overhead of thread management is just bad
  */
  for (size_t i = 1; i < n - 1; i++) {
    apply_stencil_row(old.row_ptr(i), old.row_ptr(i - 1), old.row_ptr(i + 1), res.row_ptr(i), m, half, eighth);
    
    res(i, 0) = old(i, 0); // copy boundary 
    res(i, m - 1) = old(i, m - 1);
  }
  for (size_t j = 0; j < m; j++) {
    res(0, j) = old(0, j); // copy boundary
    res(n - 1, j) = old(n - 1, j);
  }
}

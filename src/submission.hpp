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
void apply_stencil(const Grid& old, Grid& res) {
  size_t n = old.get_rows();
  size_t m = old.get_cols();

  res = old;

  if (n < 3 || m < 3) return; // no interior points to update

  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d eighth = _mm256_set1_pd(0.125);

  for (size_t i = 1; i < n - 1; i++) {
    const double* c = old.row_ptr(i);
    const double* up = old.row_ptr(i - 1);
    const double* down = old.row_ptr(i + 1);
    double* dst = res.row_ptr(i);

    size_t j = 1;
    for (; j + 4 <= m - 1; j += 4) { // run 4 at a time for interior points only
      __m256d vc = _mm256_loadu_pd(c + j);
      __m256d vup = _mm256_loadu_pd(up + j);
      __m256d vdown = _mm256_loadu_pd(down + j);
      __m256d vleft = _mm256_loadu_pd(c + j - 1);
      __m256d vright = _mm256_loadu_pd(c + j + 1);

      __m256d neighbors = _mm256_add_pd(_mm256_add_pd(vup, vdown), _mm256_add_pd(vleft, vright));
      __m256d r = _mm256_add_pd(_mm256_mul_pd(half, vc), _mm256_mul_pd(eighth, neighbors));
      _mm256_storeu_pd(dst + j, r);
    }

    // scalar for leftover
    for (; j < m - 1; j++) {
      dst[j] = 0.5 * c[j] + 0.125 * (up[j] + down[j] + c[j - 1] + c[j + 1]);
    }
  }
}

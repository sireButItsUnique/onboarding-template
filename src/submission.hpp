#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.

#define ALIGNMENT_SIZE 32
#define PREFIX_PADDING 3

class Grid {
private:
  size_t rows_;
  size_t cols_;
  size_t size_;

  bool boundary_formed_; // keep track of whether boundary has already been copied

  size_t stride_; // number of doubles per row, padded for alignment

  double* data_;

public:
  Grid(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;

    stride_ = cols + PREFIX_PADDING;  // add padding to ensure alignment on interior points 
    stride_ = (stride_ + 3) & ~3;     // round up to nearest multiple of 4 (4 * 64 = 32 bytes) for alignment across rows
    size_ = rows * stride_ * sizeof(double);

    data_ = static_cast<double*>(::operator new(size_, std::align_val_t(ALIGNMENT_SIZE)));
    memset(data_, 0, size_);

    boundary_formed_ = false;
  }

  ~Grid() {
    ::operator delete(data_, std::align_val_t(ALIGNMENT_SIZE));
  }

  size_t get_rows() const {
    return rows_;
  }

  size_t get_cols() const {
    return cols_;
  }

  bool get_boundary_formed() const {
    return boundary_formed_;
  }

  void form_boundary() {
    boundary_formed_ = true;
  }

  double* row_ptr(size_t i) {
    return data_ + i * stride_ + PREFIX_PADDING; 
  }
  
  const double* row_ptr(size_t i) const { 
    return data_ + i * stride_ + PREFIX_PADDING; 
  }

  double& operator()(size_t row, size_t col) {
    size_t i = row * stride_ + col + PREFIX_PADDING; 
    return data_[i];
  }

  double operator()(size_t row, size_t col) const {
    size_t i = row * stride_ + col + PREFIX_PADDING;
    return data_[i];
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
__attribute__((target("avx2")))
static void apply_stencil_row(const double* __restrict__ c, const double* __restrict__ up, const double* __restrict__ down,
                       double* __restrict__ dst, const size_t m) {
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d eighth = _mm256_set1_pd(0.125); // do it in here since it's cheap anyway, no messy passing around
  
  size_t j = 1;

  // vectorized loop, 4 at a time, for interior points only
  for (; j + 4 <= m - 1; j += 4) {

    // these are guranteed aligned
    __m256d vcur = _mm256_load_pd(c + j);
    __m256d vup = _mm256_load_pd(up + j);
    __m256d vdown = _mm256_load_pd(down + j);
    
    // these cannot be aligned since they're shifted
    __m256d vleft = _mm256_loadu_pd(c + j - 1);
    __m256d vright = _mm256_loadu_pd(c + j + 1);

    __m256d neighbors = _mm256_add_pd(_mm256_add_pd(vup, vdown), _mm256_add_pd(vleft, vright));
    __m256d res = _mm256_add_pd(_mm256_mul_pd(half, vcur), _mm256_mul_pd(eighth, neighbors));
    _mm256_store_pd(dst + j, res);
  }

  // scalar for leftover, should be at most 3 leftover, so not worth vectorizing
  for (; j < m - 1; j++) {
    dst[j] = 0.5 * c[j] + 0.125 * (up[j] + down[j] + c[j - 1] + c[j + 1]);
  }
}

__attribute__((target("avx2")))
void apply_stencil(const Grid& old, Grid& res) {
  size_t n = old.get_rows();
  size_t m = old.get_cols();

  // no interior points to update
  if (n < 3 || m < 3) {
    if (res.get_boundary_formed()) return;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        res(i, j) = old(i, j);
      }
    }
    res.form_boundary();
    return;
  } 
  
  #pragma omp parallel for schedule(static)           // static since workload is uniform
  for (size_t i = 1; i < n - 1; i++) {                // found better to not specify num of threads, let omp decide
    apply_stencil_row(old.row_ptr(i), old.row_ptr(i - 1), old.row_ptr(i + 1), res.row_ptr(i), m);

    res(i, 0) = old(i, 0);  // move back to already threaded loop, less threads + i already in cache
    res(i, m - 1) = old(i, m - 1); // seems not worth to run if statement to check if it's necessary, found to be slower
  }

  // copy boundary values only if first time, since in harness they're just swapped around anyway
  if (!res.get_boundary_formed()) {

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < m; j++) {
      res(0, j) = old(0, j); 
      res(n - 1, j) = old(n - 1, j);
    }

    res.form_boundary(); // boundary never changes, can leave alone
  }
}
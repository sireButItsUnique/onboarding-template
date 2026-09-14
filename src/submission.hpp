#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

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

  Grid(const Grid& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    size_ = other.size_;
    
    data_ = (double*)malloc(size_);
    memcpy(data_, other.data_, size_);
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
void apply_stencil(const Grid& old, Grid& res) {
  size_t n = old.get_rows();
  size_t m = old.get_cols();

  res = old;
  for (int i = 1; i < n - 1; i++) {
    for (int j = 1; j < m - 1; j++) {
      res(i, j) = 0.5 * old(i, j);
      res(i, j) += 0.125 * (old(i + 1, j) + old(i - 1, j) + old(i, j + 1) + old(i, j - 1));
    }
  }
}

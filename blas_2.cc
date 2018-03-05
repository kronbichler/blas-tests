
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>


// -----------------------------------------------------------------------------
void do_output (const std::size_t size,
                const std::size_t n_repetitions,
                const double      time)
{
  std::cout << std::setw(12) << 1e-9 * size * n_repetitions * 2. / time
            << " GFLOPs/s, "
            << std::setw(12) << 1e-9 * size * n_repetitions * 8. / time
            << " GB/s"
            << std::endl;
}



// -----------------------------------------------------------------------------
void run_dgemv (const std::size_t m,
                const std::size_t n,
                const std::size_t n_repetitions)
{
  std::cout << "Matrix size " << m << " x " << n << std::endl;

  std::vector<double> x(n), y(m), A(n*m);
  const double a = static_cast<double>(rand())/RAND_MAX;
  for (std::size_t i=0; i<n; ++i)
    x[i] = static_cast<double>(rand())/RAND_MAX;
  for (std::size_t i=0; i<m*n; ++i)
    A[i] = static_cast<double>(rand())/RAND_MAX;

  std::cout << "Test y = A * x" << std::endl;

  auto tinit = std::chrono::high_resolution_clock::now();
  for (std::size_t t=0; t<n_repetitions; ++t)
    for (std::size_t i=0; i<m; ++i)
      {
        double sum = 0;
        for (std::size_t j=0; j<n; ++j)
          sum += A[i+j*m] * x[j];
        y[i] += sum;
      }

  std::cout << "Serial  1 thread  ";
  do_output(n*m, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count());

  double check = 0;
  for (unsigned int i=0; i<m; ++i)
    check += std::abs(y[i]);

  for (unsigned int i=0; i<m; ++i)
    y[i] = 0.;

  tinit = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(none) shared (x, y, A)
  {
    for (std::size_t t=0; t<n_repetitions; ++t)
#pragma omp for schedule (static)
      for (std::size_t i=0; i<m; ++i)
        {
          double sum = 0;
          for (std::size_t j=0; j<n; ++j)
            sum += A[i+j*m] * x[j];
          y[i] += sum;
        }
  }

  std::cout << "OpenMP " << std::setw(2) << omp_get_max_threads() << " threads ";
  do_output(n*m, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count());

  double check2 = 0;
  for (unsigned int i=0; i<m; ++i)
    check2 += std::abs(y[i]);
  if (std::abs(check - check2) > 1e-8 * check)
    std::cout << "Parallel loop result is not correct: "
              << check << " vs " << check2 << std::endl;


  std::cout << "Test x = A^T * y" << std::endl;

  tinit = std::chrono::high_resolution_clock::now();
  for (std::size_t t=0; t<n_repetitions; ++t)
    for (std::size_t j=0; j<n; ++j)
      {
        double sum = 0;
        for (std::size_t i=0; i<m; ++i)
          sum += A[i+j*m] * y[i];
        x[j] += sum;
      }

  std::cout << "Serial  1 thread  ";
  do_output(n*m, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count());

  check = 0;
  for (unsigned int i=0; i<m; ++i)
    check += std::abs(y[i]);

  for (unsigned int i=0; i<m; ++i)
    x[i] = 0.;

  tinit = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(none) shared (x, y, A)
  {
    for (std::size_t t=0; t<n_repetitions; ++t)
#pragma omp for schedule (static)
      for (std::size_t j=0; j<n; ++j)
        {
          double sum = 0;
          for (std::size_t i=0; i<m; ++i)
            sum += A[i+j*m] * y[i];
          x[j] += sum;
        }
  }

  std::cout << "OpenMP " << std::setw(2) << omp_get_max_threads() << " threads ";
  do_output(n*m, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count());

  check2 = 0;
  for (unsigned int i=0; i<m; ++i)
    check2 += std::abs(y[i]);
  if (std::abs(check - check2) > 1e-8 * check)
    std::cout << "Parallel loop result is not correct: "
              << check << " vs " << check2 << std::endl;
}



// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  int m = 1000;
  if (argc > 1)
    m = std::atoi(argv[1]);
  int n = m;
  if (argc > 2)
    n = std::atoi(argv[2]);
  std::size_t n_repetitions = 1000000000ULL / (n*m);
  if (argc > 3)
    n_repetitions = std::atoi(argv[3]);

  run_dgemv(m, n, n_repetitions);

  return 0;
}

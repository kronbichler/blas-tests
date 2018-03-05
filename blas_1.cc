
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>


// -----------------------------------------------------------------------------
void do_output (const std::size_t size,
                const std::size_t n_repetitions,
                const double      time,
                const std::size_t ops_per_entry,
                const std::size_t mem_per_entry)
{
  std::cout << std::setw(12) << 1e-6 * size * n_repetitions / time
            << " million entries per second, "
            << std::setw(12) << 1e-9 * size * n_repetitions * ops_per_entry / time
            << " GFLOPs/s, "
            << std::setw(12) << 1e-9 * size*n_repetitions * mem_per_entry / time
            << " GB/s"
            << std::endl;
}



// -----------------------------------------------------------------------------
void run_daxpy (const std::size_t size,
                const std::size_t n_repetitions)
{
  std::cout << "Vector size is " << size << std::endl;

  std::vector<double> x(size), y(size), z(size);
  const double a = static_cast<double>(rand())/RAND_MAX;
  for (std::size_t i=0; i<size; ++i)
    x[i] = static_cast<double>(rand())/RAND_MAX;

  std::cout << "Test z = a * x + y" << std::endl;

  auto tinit = std::chrono::high_resolution_clock::now();
  for (std::size_t t=0; t<n_repetitions; ++t)
    for (std::size_t i=0; i<size; ++i)
      z[i] = a*x[i] + y[i];

  std::cout << "Serial  1 thread  ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8. + 8. + 16.);


  tinit = std::chrono::high_resolution_clock::now();

#pragma omp parallel default(none) shared (x, y, z)
  {
    for (std::size_t t=0; t<n_repetitions; ++t)
#pragma omp for schedule (static)
      for (std::size_t i=0; i<size; ++i)
        z[i] = a*x[i] + y[i];
  }

  std::cout << "OpenMP " << std::setw(2) << omp_get_max_threads() << " threads ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8. + 8. + 16.);
}



// -----------------------------------------------------------------------------
void run_ddot (const std::size_t size,
               const std::size_t n_repetitions)
{
  std::cout << "Vector size is " << size << std::endl;

  std::vector<double> x(size), y(size);
  const double a = static_cast<double>(rand())/RAND_MAX;
  for (std::size_t i=0; i<size; ++i)
    x[i] = static_cast<double>(rand())/RAND_MAX;
  for (std::size_t i=0; i<size; ++i)
    y[i] = static_cast<double>(rand())/RAND_MAX;

  std::cout << "Test a = x^T * y" << std::endl;
  double sum = 0;

  auto tinit = std::chrono::high_resolution_clock::now();
  for (std::size_t t=0; t<n_repetitions; ++t)
    for (std::size_t i=0; i<size; ++i)
      sum += x[i] * y[i];

  std::cout << "Serial  1 thread  ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8. + 8.);


  tinit = std::chrono::high_resolution_clock::now();

  double sum2 = 0;
#pragma omp parallel default(none) shared (x, y, sum2)
  {
    double my_sum = 0;
    for (std::size_t t=0; t<n_repetitions; ++t)
#pragma omp for schedule (static)
      for (std::size_t i=0; i<size; ++i)
        my_sum += x[i] * y[i];

#pragma omp critical
    sum2 += my_sum;
  }

  std::cout << "OpenMP " << std::setw(2) << omp_get_max_threads() << " threads ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8. + 8.);

  if (std::abs(sum - sum2) > 1e-8 * sum)
    std::cout << "Parallel loop result is not correct: "
              << sum << " vs " << sum2 << std::endl;


  double sum3 = 0;

  tinit = std::chrono::high_resolution_clock::now();
  for (std::size_t t=0; t<n_repetitions; ++t)
    {
      double local_sums[10];
      for (unsigned int i=0; i<10; ++i)
        local_sums[i] = 0;
      const std::size_t size_threshold = (size / 10) * 10;
      for (std::size_t i=0; i<size_threshold; i+=10)
        for (unsigned int j=0; j<10; ++j)
          local_sums[j] += x[i+j] * y[i+j];
      for (std::size_t i=size_threshold; i<size; ++i)
        local_sums[0] += x[i] * y[i];
      for (unsigned int j=0; j<10; ++j)
        sum3 += local_sums[j];
    }

  std::cout << "Serial reordered  ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8. + 8.);

  if (std::abs(sum - sum3) > 1e-8 * sum)
    std::cout << "Serial reordered loop result is not correct: "
              << sum << " vs " << sum3 << std::endl;
}


// -----------------------------------------------------------------------------
void run_norm2 (const std::size_t size,
                const std::size_t n_repetitions)
{
  std::cout << "Vector size is " << size << std::endl;

  std::vector<double> x(size);
  const double a = static_cast<double>(rand())/RAND_MAX;
  for (std::size_t i=0; i<size; ++i)
    x[i] = static_cast<double>(rand())/RAND_MAX;

  std::cout << "Test a = x^T * x" << std::endl;
  double sum = 0;

  auto tinit = std::chrono::high_resolution_clock::now();
  for (std::size_t t=0; t<n_repetitions; ++t)
    for (std::size_t i=0; i<size; ++i)
      sum += x[i] * x[i];

  std::cout << "Serial  1 thread  ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8.);

  tinit = std::chrono::high_resolution_clock::now();

  double sum2 = 0;
#pragma omp parallel default(none) shared (x, sum2)
  {
    double my_sum = 0;
    for (std::size_t t=0; t<n_repetitions; ++t)
#pragma omp for schedule (static)
      for (std::size_t i=0; i<size; ++i)
        my_sum += x[i] * x[i];

#pragma omp critical
    sum2 += my_sum;
  }

  std::cout << "OpenMP " << std::setw(2) << omp_get_max_threads() << " threads ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8.);

  if (std::abs(sum - sum2) > 1e-8 * sum)
    std::cout << "Parallel loop result is not correct: "
              << sum << " vs " << sum2 << std::endl;


  double sum3 = 0;

  tinit = std::chrono::high_resolution_clock::now();
  for (std::size_t t=0; t<n_repetitions; ++t)
    {
      double local_sums[10];
      for (unsigned int i=0; i<10; ++i)
        local_sums[i] = 0;
      const std::size_t size_threshold = (size / 10) * 10;
      for (std::size_t i=0; i<size_threshold; i+=10)
        for (unsigned int j=0; j<10; ++j)
          local_sums[j] += x[i+j] * x[i+j];
      for (std::size_t i=size_threshold; i<size; ++i)
        local_sums[0] += x[i] * x[i];
      for (unsigned int j=0; j<10; ++j)
        sum3 += local_sums[j];
    }

  std::cout << "Serial reordered  ";
  do_output(size, n_repetitions,
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-tinit).count(),
            2., 8.);

  if (std::abs(sum - sum3) > 1e-8 * sum)
    std::cout << "Serial reordered loop result is not correct: "
              << sum << " vs " << sum3 << std::endl;
}



// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  int size = 1000;
  if (argc > 1)
    size = std::atoi(argv[1]);
  std::size_t n_repetitions = 1000000000ULL / size;
  if (argc > 2)
    n_repetitions = std::atoi(argv[2]);

  run_daxpy(size, n_repetitions);
  run_ddot(size, n_repetitions);
  run_norm2(size, n_repetitions);

  return 0;
}

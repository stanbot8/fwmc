#ifndef FWMC_PLATFORM_H_
#define FWMC_PLATFORM_H_

// MSVC uses __restrict, GCC/Clang use __restrict__
#ifdef _MSC_VER
  #define FWMC_RESTRICT __restrict
#else
  #define FWMC_RESTRICT __restrict__
#endif

#endif  // FWMC_PLATFORM_H_

set(LIBC_FLAG_LIST "")

# Define flags and their options
list(APPEND LIBC_FLAG_LIST "X86")
set(FLAG_X86_OPTIONS "SSE2;SSE4.1;SSE4.2;AVX;AVX2;FMA;AVX512F;AVX512BW")

if(NOT DEFINED FLAG_X86_DEFAULT)
  # What should be the default value.
endif()

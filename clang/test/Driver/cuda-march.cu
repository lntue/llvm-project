// Checks that cuda compilation does the right thing when passed -march.
// (Specifically, we want to pass it to host compilation, but not to device
// compilation or ptxas!)

// RUN: %clang -### --target=x86_64-linux-gnu -c \
// RUN: -nogpulib -nogpuinc -march=haswell %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-linux-gnu -c \
// RUN: -nogpulib -nogpuinc -march=haswell --cuda-gpu-arch=sm_52 %s 2>&1 | FileCheck %s

// CHECK: "-cc1"{{.*}} "-triple" "nvptx
// CHECK-SAME: "-target-cpu" "sm_52"

// CHECK: ptxas
// CHECK-SAME: "--gpu-name" "sm_52"

// CHECK: "-cc1"{{.*}} "-target-cpu" "haswell"

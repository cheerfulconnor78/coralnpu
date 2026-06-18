#include <stdint.h>
#include <stddef.h>
#include "sw/utils/utils.h"

// Reduced to 1024 elements to match the Python test script.
// 1024 floats = 4 KB per buffer (8 KB total), which easily fits in the 64 KB DTCM.
constexpr size_t kElements = 1024; 

// Input and Output SRAM buffers (names must match exactly what Cocotb looks up)
float in_buf[kElements] __attribute__((section(".data"), used, retain)) __attribute__((aligned(16)));
float out_buf[kElements] __attribute__((section(".data"), used, retain)) __attribute__((aligned(16)));

extern "C" {
  volatile uint32_t csr_cycle_count = 0;
}

extern "C" void GeluRVV(const float* I, float* O, size_t num_elements);

int main(int argc, char** argv) {
  // Enable cycle counting CSRs
  uint32_t mcontext0_write_value = 1;
  asm volatile("csrw 0x7C0, %0" : : "r"(mcontext0_write_value));

  cycle_counter_reset();
  uint64_t start_cycles = mcycle_read();

  // Execute GELU kernel
  GeluRVV(in_buf, out_buf, kElements);

  uint64_t end_cycles = mcycle_read();
  csr_cycle_count = static_cast<uint32_t>(end_cycles - start_cycles);

  // Disable cycle counting CSRs
  mcontext0_write_value = 0;
  asm volatile("csrw 0x7C0, %0" : : "r"(mcontext0_write_value));

  return 0;
}

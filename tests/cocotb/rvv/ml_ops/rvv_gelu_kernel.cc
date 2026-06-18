#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

// Reverted to LMUL=8 to maximize vector ALU occupancy and minimize instruction fetch
inline vfloat32m8_t rvv_exp_f32m8(vfloat32m8_t x, size_t vl) {
    vfloat32m8_t v_log2e = __riscv_vfmv_v_f_f32m8(1.44269504f, vl);
    vfloat32m8_t v_ln2   = __riscv_vfmv_v_f_f32m8(0.69314718f, vl);
    vfloat32m8_t v_one   = __riscv_vfmv_v_f_f32m8(1.0f, vl);
    vfloat32m8_t v_min   = __riscv_vfmv_v_f_f32m8(-88.0f, vl);

    x = __riscv_vfmax_vv_f32m8(x, v_min, vl);
    vfloat32m8_t y = __riscv_vfmul_vv_f32m8(x, v_log2e, vl);
    vint32m8_t   i_int   = __riscv_vfcvt_x_f_v_i32m8(y, vl);
    vfloat32m8_t i_float = __riscv_vfcvt_f_x_v_f32m8(i_int, vl);
    vfloat32m8_t i_ln2   = __riscv_vfmul_vv_f32m8(i_float, v_ln2, vl);
    vfloat32m8_t f       = __riscv_vfsub_vv_f32m8(x, i_ln2, vl);

    vfloat32m8_t p;
    vfloat32m8_t c2 = __riscv_vfmv_v_f_f32m8(0.5f, vl);
    vfloat32m8_t c3 = __riscv_vfmv_v_f_f32m8(0.16666667f, vl);
    p = __riscv_vfmacc_vv_f32m8(c2, f, c3, vl);
    p = __riscv_vfmacc_vv_f32m8(v_one, f, p, vl);
    p = __riscv_vfmacc_vv_f32m8(v_one, f, p, vl);

    vint32m8_t bias      = __riscv_vmv_v_x_i32m8(127, vl);
    vint32m8_t exp_bits  = __riscv_vadd_vv_i32m8(i_int, bias, vl);
    exp_bits             = __riscv_vsll_vx_i32m8(exp_bits, 23, vl);
    vfloat32m8_t v_scale = __riscv_vreinterpret_v_i32m8_f32m8(exp_bits);
    return __riscv_vfmul_vv_f32m8(p, v_scale, vl);
}

extern "C" void GeluRVV(const float* I, float* O, size_t num_elements) {
    // Hoist constants to maximum vector length
    size_t max_vl = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t v_c1  = __riscv_vfmv_v_f_f32m8(1.59576912f, max_vl);
    vfloat32m8_t v_c2  = __riscv_vfmv_v_f_f32m8(0.071354816f, max_vl);
    vfloat32m8_t v_one = __riscv_vfmv_v_f_f32m8(1.0f, max_vl);
    vfloat32m8_t v_two = __riscv_vfmv_v_f_f32m8(2.0f, max_vl);

    size_t vl;
    for (size_t i = 0; i < num_elements; i += vl) {
        vl = __riscv_vsetvl_e32m8(num_elements - i);

        vfloat32m8_t v_x = __riscv_vle32_v_f32m8(I + i, vl);

        // Polynomial
        vfloat32m8_t v_x2 = __riscv_vfmul_vv_f32m8(v_x, v_x, vl);
        vfloat32m8_t v_x3 = __riscv_vfmul_vv_f32m8(v_x, v_x2, vl);
        
        vfloat32m8_t v_y = __riscv_vfmul_vv_f32m8(v_x, v_c1, vl);
        v_y = __riscv_vfmacc_vv_f32m8(v_y, v_c2, v_x3, vl);

        // Exponential
        vfloat32m8_t v_neg_y = __riscv_vfneg_v_f32m8(v_y, vl);
        vfloat32m8_t v_exp = rvv_exp_f32m8(v_neg_y, vl);
        vfloat32m8_t v_d = __riscv_vfadd_vv_f32m8(v_exp, v_one, vl);

        // Newton-Raphson Fast Reciprocal (replaces vfdiv)
        vfloat32m8_t v_rec = __riscv_vfrec7_v_f32m8(v_d, vl);
        vfloat32m8_t v_err = __riscv_vfnmsac_vv_f32m8(v_two, v_d, v_rec, vl); 
        v_rec = __riscv_vfmul_vv_f32m8(v_rec, v_err, vl);

        // Output = x * (1/d)
        vfloat32m8_t v_out = __riscv_vfmul_vv_f32m8(v_x, v_rec, vl);
        __riscv_vse32_v_f32m8(O + i, v_out, vl);
    }
}

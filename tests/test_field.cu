/*
 * test_field.cu — Field arithmetic correctness tests
 *
 * Verifies field operations against known test vectors.
 * Run: make test
 */

#include <stdio.h>
#include <string.h>
#include "field.cuh"
#include "ec.cuh"

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

static void print_u256(const char *label, const u256 *v) {
    printf("  %s: %016lx %016lx %016lx %016lx\n",
           label, v->d[3], v->d[2], v->d[1], v->d[0]);
}

static void check(const char *name, int cond) {
    if (cond) { tests_passed++; printf("  [PASS] %s\n", name); }
    else      { tests_failed++; printf("  [FAIL] %s\n", name); }
}

// GPU kernel that runs tests and writes results to device memory
__global__ void test_kernel(int *results) {
    int idx = 0;

    // Test 1: fp_add identity
    {
        u256 zero = {{0, 0, 0, 0}};
        u256 one = {{1, 0, 0, 0}};
        u256 r = fp_add(&zero, &one);
        results[idx++] = (r.d[0] == 1 && r.d[1] == 0 && r.d[2] == 0 && r.d[3] == 0);
    }

    // Test 2: fp_add wraps mod P
    {
        u256 p_minus_1 = SECP256K1_P;
        p_minus_1.d[0] -= 1;
        u256 one = {{1, 0, 0, 0}};
        u256 r = fp_add(&p_minus_1, &one);
        results[idx++] = u256_is_zero(&r);
    }

    // Test 3: fp_sub basic
    {
        u256 five = {{5, 0, 0, 0}};
        u256 three = {{3, 0, 0, 0}};
        u256 r = fp_sub(&five, &three);
        results[idx++] = (r.d[0] == 2 && r.d[1] == 0 && r.d[2] == 0 && r.d[3] == 0);
    }

    // Test 4: fp_sub wraps (0 - 1 = P - 1)
    {
        u256 zero = {{0, 0, 0, 0}};
        u256 one = {{1, 0, 0, 0}};
        u256 r = fp_sub(&zero, &one);
        u256 expected = SECP256K1_P;
        expected.d[0] -= 1;
        results[idx++] = u256_eq(&r, &expected);
    }

    // Test 5: fp_mul by 1
    {
        u256 val = {{0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0, 0}};
        u256 one = {{1, 0, 0, 0}};
        u256 r = fp_mul(&val, &one);
        results[idx++] = u256_eq(&r, &val);
    }

    // Test 6: fp_mul by 0
    {
        u256 val = {{0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0, 0}};
        u256 zero = {{0, 0, 0, 0}};
        u256 r = fp_mul(&val, &zero);
        results[idx++] = u256_is_zero(&r);
    }

    // Test 7: fp_sqr(2) = 4
    {
        u256 two = {{2, 0, 0, 0}};
        u256 r = fp_sqr(&two);
        results[idx++] = (r.d[0] == 4 && r.d[1] == 0 && r.d[2] == 0 && r.d[3] == 0);
    }

    // Test 8: fp_mul commutativity
    {
        u256 a = {{7, 0, 0, 0}};
        u256 b = {{11, 0, 0, 0}};
        u256 ab = fp_mul(&a, &b);
        u256 ba = fp_mul(&b, &a);
        results[idx++] = u256_eq(&ab, &ba);
    }

    // Test 9: 7 * 11 = 77
    {
        u256 a = {{7, 0, 0, 0}};
        u256 b = {{11, 0, 0, 0}};
        u256 r = fp_mul(&a, &b);
        results[idx++] = (r.d[0] == 77 && r.d[1] == 0 && r.d[2] == 0 && r.d[3] == 0);
    }

    // Test 10: fp_inv(1) = 1
    {
        u256 one = {{1, 0, 0, 0}};
        u256 r = fp_inv(&one);
        results[idx++] = (r.d[0] == 1 && r.d[1] == 0 && r.d[2] == 0 && r.d[3] == 0);
    }

    // Test 11: fp_inv(a) * a = 1
    {
        u256 a = {{0xDEADBEEFCAFEBABEULL, 0x1234567890ABCDEFULL, 0, 0}};
        u256 a_inv = fp_inv(&a);
        u256 r = fp_mul(&a, &a_inv);
        results[idx++] = (r.d[0] == 1 && r.d[1] == 0 && r.d[2] == 0 && r.d[3] == 0);
    }

    // Test 12: Generator point is on curve (y^2 = x^3 + 7)
    {
        u256 y2 = fp_sqr(&GENERATOR.y);
        u256 x2 = fp_sqr(&GENERATOR.x);
        u256 x3 = fp_mul(&x2, &GENERATOR.x);
        u256 seven = {{7, 0, 0, 0}};
        u256 rhs = fp_add(&x3, &seven);
        results[idx++] = u256_eq(&y2, &rhs);
    }

    // Test 13: 2*G via ec_double_j gives correct affine point
    // 2*G.x = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
    {
        JacobianPoint G_jac;
        G_jac.X = GENERATOR.x; G_jac.Y = GENERATOR.y;
        G_jac.Z.d[0] = 1; G_jac.Z.d[1] = 0; G_jac.Z.d[2] = 0; G_jac.Z.d[3] = 0;
        JacobianPoint G2 = ec_double_j(&G_jac);
        AffinePoint G2a = ec_to_affine(&G2);
        // Check x-coordinate of 2*G
        u256 expected_x = {{0xABAC09B95C709EE5ULL, 0x5C778E4B8CEF3CA7ULL,
                            0x3045406E95C07CD8ULL, 0xC6047F9441ED7D6DULL}};
        results[idx++] = u256_eq(&G2a.x, &expected_x);
    }

    // Test 14: ec_scalar_mul(2, G) matches ec_double_j(G)
    {
        u256 two = {{2, 0, 0, 0}};
        JacobianPoint sG = ec_scalar_mul(&two, &GENERATOR);
        AffinePoint sGa = ec_to_affine(&sG);
        u256 expected_x = {{0xABAC09B95C709EE5ULL, 0x5C778E4B8CEF3CA7ULL,
                            0x3045406E95C07CD8ULL, 0xC6047F9441ED7D6DULL}};
        results[idx++] = u256_eq(&sGa.x, &expected_x);
    }

    // Store total test count
    results[99] = idx;
}

int main() {
    printf("\n  HYDRA KANGAROO — Field & EC Arithmetic Tests\n");
    printf("  =============================================\n\n");

    int *d_results;
    int h_results[100] = {0};
    cudaMalloc(&d_results, 100 * sizeof(int));
    cudaMemset(d_results, 0, 100 * sizeof(int));

    test_kernel<<<1, 1>>>(d_results);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_results, d_results, 100 * sizeof(int), cudaMemcpyDeviceToHost);

    int num_tests = h_results[99];
    const char *test_names[] = {
        "fp_add: 0 + 1 = 1",
        "fp_add: (P-1) + 1 = 0",
        "fp_sub: 5 - 3 = 2",
        "fp_sub: 0 - 1 = P - 1",
        "fp_mul: a * 1 = a",
        "fp_mul: a * 0 = 0",
        "fp_sqr: 2^2 = 4",
        "fp_mul: commutativity",
        "fp_mul: 7 * 11 = 77",
        "fp_inv: inv(1) = 1",
        "fp_inv: inv(a) * a = 1",
        "EC: Generator on curve",
        "EC: double(G) correct x",
        "EC: scalar_mul(2,G) matches double(G)",
    };

    for (int i = 0; i < num_tests && i < 14; i++) {
        check(test_names[i], h_results[i]);
    }

    printf("\n  Results: %d passed, %d failed out of %d tests\n\n",
           tests_passed, tests_failed, num_tests);

    cudaFree(d_results);
    return tests_failed > 0 ? 1 : 0;
}

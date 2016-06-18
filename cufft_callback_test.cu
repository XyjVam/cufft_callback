#include <iostream>
#include "cufft_callback.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <criterion/criterion.h>
#include <criterion/logging.h>

using namespace cufftcallback;

// used with custom type
/********************************/
/* SCALE USING A CUFFT CALLBACK */
/********************************/

struct Scale
{
  enum {eParamNum = 0};
  __device__ float2 operator()(ParPos<0> l,float2& e,size_t of,void* i)
  {
    return e*0.25f;
  }
};

#define CALL_FFT() \
  checkCudaErrors(cufftExecC2C(plan, thrust::raw_pointer_cast(d_vec.data()) \
        ,thrust::raw_pointer_cast(d_vec.data()), CUFFT_FORWARD));

/*
 * Test: Using of a custom functor to scale
 */
// Register callback
CUFFT_STORE_CALLBACK_REG_CUSTOM(scale_025,Scale);
Test(cufft_callbacks,custom_scale)
{
  const int N=48;
  // --- Setting up input device vector
  thrust::device_vector<float2> d_vec;
  cr_assert_none_throw(d_vec.resize(N,make_cuComplex(1.0f,2.0f)));
  // --- Callback initialization
  CUFFT_STORE_CALLBACK_INIT_CUSTOM(scale_025,Scale);
  //Create cufft plan
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_C2C, 1);
  // --- Apply the callback
  cuFftStoreCallback::apply<Scale>(plan);
  // --- Perform in-place direct Fourier transform
  CALL_FFT();
  // --- Setting up output host vector
  thrust::host_vector<float2> h_vec;
  cr_expect_none_throw(h_vec.resize(N));
  cr_expect_none_throw(h_vec = d_vec);
  cr_expect_eq(h_vec[0], make_cuComplex(12.,24.));
  cr_expect_eq(h_vec[1], make_cuComplex(0.0f,0.0f));
  //Clean up
  checkCudaErrors(cufftDestroy(plan));
}

/*
 * Test: multiply with an array of floats [*=f_i]
 */
// Register callback
CUFFT_STORE_CALLBACK_REG(mul_float,CFloat()*=FloatUser());
Test(cufft_callbacks,mul_float)
{
  const int N=48;
  // --- Setting up input device vector
  thrust::device_vector<float2> d_vec;
  cr_assert_none_throw(d_vec.resize(N,make_cuComplex(1.0f,2.0f)));
  // --- Setting up scaling device vector
  thrust::device_vector<float> d_vecSc;
  cr_assert_none_throw(d_vecSc.resize(N,0.5f));
  // --- Callback initialization
  CUFFT_STORE_CALLBACK_INIT(mul_float);
  //Create cufft plan
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_C2C, 1);
  // --- Apply the callback
  cuFftStoreCallback::apply<REG_TYPE(mul_float)>
  (plan,thrust::raw_pointer_cast(d_vecSc.data()));
  // --- Perform in-place direct Fourier transform
  CALL_FFT();
  // --- Setting up output host vector
  thrust::host_vector<float2> h_vec(N);
  h_vec = d_vec;
  cr_expect_eq(h_vec[0], make_cuComplex(24.,48.));
  cr_expect_eq(h_vec[1], make_cuComplex(0.0f,0.0f));
  //Clean up
  checkCudaErrors(cufftDestroy(plan));
}

/*
 * Test: multiply with an array of complex floats [*=c_i]
 */
CUFFT_STORE_CALLBACK_REG(mul_cf,CFloat()*=CFloatUser());
Test(cufft_callbacks,mul_cf)
{
  const int N=48;
  // --- Setting up input device vector
  thrust::device_vector<float2> d_vec;
  cr_assert_none_throw(d_vec.resize(N,make_cuComplex(1.0f,2.0f)));
  // --- Setting up scaling device vectors
  thrust::device_vector<float2> d_vecC;
  cr_assert_none_throw(d_vecC.resize(N,make_cuComplex(0.01f,0.1f)));
  // --- Callback initialization
  CUFFT_STORE_CALLBACK_INIT(mul_cf);
  //Create cufft plan
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_C2C, 1);
  // --- Apply the callback
  cuFftStoreCallback::apply<REG_TYPE(mul_cf)>
  (plan,thrust::raw_pointer_cast(d_vecC.data()));
  // --- Perform in-place direct Fourier transform
  CALL_FFT();
  // --- Setting up output host vector
  thrust::host_vector<float2> h_vec(N);
  h_vec = d_vec;
  //criterion_info(" (%f,%f)\n", h_vec[0].x,h_vec[0].y);
  cr_expect_eq(h_vec[0], make_cuComplex(-9.12f,5.76f));
  cr_expect_eq(h_vec[1], make_cuComplex(0.0f,0.0f));
  //Clean up
  checkCudaErrors(cufftDestroy(plan));
}

/*
 * Test: multiply with itself [l*=l]
 */
CUFFT_STORE_CALLBACK_REG(mul_l,CFloat()*=CFloat());
Test(cufft_callbacks,mul_l)
{
  const int N=48;
  // --- Setting up input device vector
  thrust::device_vector<float2> d_vec;
  cr_assert_none_throw(d_vec.resize(N,make_cuComplex(1.0f,2.0f)));
  // --- Callback initialization
  CUFFT_STORE_CALLBACK_INIT(mul_l);
  //Create cufft plan
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_C2C, 1);
  // --- Apply the callback
  cuFftStoreCallback::apply<REG_TYPE(mul_l)>
  (plan);
  // --- Perform in-place direct Fourier transform
  CALL_FFT();
  // --- Setting up output host vector
  thrust::host_vector<float2> h_vec(N);
  h_vec = d_vec;
  cr_expect_eq(h_vec[0], make_cuComplex(-6912.f,9216.f));
  cr_expect_eq(h_vec[1], make_cuComplex(0.0f,0.0f));
  //Clean up
  checkCudaErrors(cufftDestroy(plan));
}

/*
 * Test: *=f1_i-f2_i
 */
CUFFT_STORE_CALLBACK_REG(mul_f1_MINUS_f2,CFloat()*=FloatUser()-FloatUser());
Test(cufft_callbacks,mul_float_minus_float)
{
  const int N=48;
  // --- Setting up input device vector
  thrust::device_vector<float2> d_vec;
  cr_assert_none_throw(d_vec.resize(N,make_cuComplex(1.0f,2.0f)));
  // --- Setting up scaling device vector
  thrust::device_vector<float> d_vecSc1;
  cr_assert_none_throw(d_vecSc1.resize(N,0.2f));
  thrust::device_vector<float> d_vecSc2;
  cr_assert_none_throw(d_vecSc2.resize(N,0.1f));
  // --- Callback initialization
  CUFFT_STORE_CALLBACK_INIT(mul_f1_MINUS_f2);
  //Create cufft plan
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_C2C, 1);
  // --- Apply the callback
  cuFftStoreCallback::apply<REG_TYPE(mul_f1_MINUS_f2)>
  (plan,thrust::raw_pointer_cast(d_vecSc1.data())
     ,thrust::raw_pointer_cast(d_vecSc2.data()));
  // --- Perform in-place direct Fourier transform
  CALL_FFT();
  // --- Setting up output host vector
  thrust::host_vector<float2> h_vec(N);
  h_vec = d_vec;
  cr_expect_eq(h_vec[0], make_cuComplex(4.8f,9.6f));
  cr_expect_eq(h_vec[1], make_cuComplex(0.0f,0.0f));
  //Clean up
  checkCudaErrors(cufftDestroy(plan));
}

/*
 * Test: *=fc*(f1_i-f2_i)
 */
CUFFT_STORE_CALLBACK_REG(mul_f1c_mul_f2_minus_f3
    ,CFloat()*=FloatUserScalar()*(FloatUser()-FloatUser()));
Test(cufft_callbacks,mul_fc_mul_float_minus_float)
{
  const int N=48;
  // --- Setting up input device vector
  thrust::device_vector<float2> d_vec;
  cr_assert_none_throw(d_vec.resize(N,make_cuComplex(1.0f,2.0f)));
  // --- Setting up scaling device vectors
  thrust::device_vector<float> d_Sc0;
  cr_assert_none_throw(d_Sc0.resize(1,0.001f));
  thrust::device_vector<float> d_vecSc1;
  cr_assert_none_throw(d_vecSc1.resize(N,150.f));
  thrust::device_vector<float> d_vecSc2;
  cr_assert_none_throw(d_vecSc2.resize(N,25.0f));
  // --- Callback initialization
  CUFFT_STORE_CALLBACK_INIT(mul_f1c_mul_f2_minus_f3);
  //Create cufft plan
  cufftHandle plan;
  cufftPlan1d(&plan, N, CUFFT_C2C, 1);
  // --- Apply the callback
  cuFftStoreCallback::apply<REG_TYPE(mul_f1c_mul_f2_minus_f3)>
  (plan,thrust::raw_pointer_cast(d_Sc0.data())
   ,thrust::raw_pointer_cast(d_vecSc1.data())
   ,thrust::raw_pointer_cast(d_vecSc2.data()));
  // --- Perform in-place direct Fourier transform
  CALL_FFT();
  // --- Setting up output host vector
  thrust::host_vector<float2> h_vec(N);
  h_vec = d_vec;
  cr_expect_eq(h_vec[0], make_cuComplex(6.0f,12.0f));
  cr_expect_eq(h_vec[1], make_cuComplex(0.0f,0.0f));
  //Clean up
  checkCudaErrors(cufftDestroy(plan));
}

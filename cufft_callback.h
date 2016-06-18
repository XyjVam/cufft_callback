// file: cufft_callback.h
// ----------------------------------------------------------------------------
// Provides facility to add callback functions to cuda fft calls
//
// Usage:
//
// - Include the header
// - Register a callback function providing either a custom functor
//   e.g.
//
// >
// > struct Scale
// > {
// >   enum {eParamNum = 0};
// >   __device__ float2 operator()(ParPos<0> l,float2& e,size_t of,void* i)
// >   {
// >     return e*0.25f;
// >   }
// > };
// > CUFFT_STORE_CALLBACK_REG_CUSTOM(scale_025,Scale);
//
// or an expresion
//
// > CUFFT_STORE_CALLBACK_REG( test
// >   , CFloat()*=FloatUserScalar()*(FloatUser()-FloatUser()));
// - Initialize callback function:
// > CUFFT_STORE_CALLBACK_INIT(test);
//
// or
//
// > CUFFT_STORE_CALLBACK_INIT_CUSTOM(scale_025,Scale);
//
// in case of the custom functor.
//
// The later should be called before applying callback
//
// - Apply a callback function before an actual FFT call
//
// > cuFftStoreCallback::apply<REG_TYPE(test)>
// >   (plan,thrust::raw_pointer_cast(d_Sc0.data())
// >     ,thrust::raw_pointer_cast(d_vecSc1.data())
// >     ,thrust::raw_pointer_cast(d_vecSc2.data()));
//
// or
//
// > cuFftStoreCallback::apply<Scale>(plan);
//
// of course the plan has to be already created
//
// Example:
//
// > #include "cufft_callback.h"
// > #include <thrust/device_vector.h>
// > // We should first register a callback function
// > CUFFT_STORE_CALLBACK_REG(mul_float,CFloat()*=FloatUser());
// > // In this example we register a callback function, that will multiply
// > //  result of an FFT call with an array of floats
// > //  Note! the user have to provide an array of the same dimension
// > const int N=48;
// > int main(void) {
// >   // --- Setting up input device vector
// >   thrust::device_vector<float2> d_vec(N,make_cuComplex(1.0f,2.0f)));
// >   // --- Setting up scaling device vector
// >   thrust::device_vector<float> d_vecSc(d_vecSc.resize(N,0.5f));
// >   // --- Callback initialization
// >   CUFFT_STORE_CALLBACK_INIT(mul_float);
// >   //Create cufft plan
// >   cufftHandle plan;
// >   cufftPlan1d(&plan, N, CUFFT_C2C, 1);
// >   // --- Apply the callback
// >   cuFftStoreCallback::apply<REG_TYPE(mul_float)>
// >   (plan,thrust::raw_pointer_cast(d_vecSc.data()));
// >   cufftExecC2C(plan, thrust::raw_pointer_cast(d_vec.data())
// >         ,thrust::raw_pointer_cast(d_vec.data()), CUFFT_FORWARD);
// >   return 0;
// > }
#ifndef CUFFT_CALLBACK_H_ZCIRVK4D
#define CUFFT_CALLBACK_H_ZCIRVK4D

#include <tuple>
#include <vector>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include "cumgpu/core/gpu/cuerror.h"

namespace cumgpu
{

static  std::vector<cufftCallbackStoreC> storeCallbackVec;
static  std::vector<cufftCallbackLoadC> loadCallbackVec;

template<int I>
struct ParPos
{
  enum {L=I};
};

template<class T>
__device__ void cufft_callback_store__(
    void *dataOut,
    size_t offset,
    float2 element,
    void *callerInfo,
    void *sharedPtr)
{
  float2 output = T()(ParPos<T::eParamNum>(),element,offset,callerInfo);

  ((float2*)dataOut)[offset] = output;
}

static cufftCallbackStoreC h_storeCallbackPtrTmp = NULL;

template<class T>
struct initCuFftStoreCallBack
{
  static int cbIdx_;
  static void** p_;
};

template<class T, int I>
struct ParSet
{
  template<class U, class...Ts>
  ParSet(U u,Ts...pars)
  {
    checkCudaErrors(cudaMemcpy(&initCuFftStoreCallBack<T>::p_[I-1], &u,
          sizeof(void*), cudaMemcpyHostToDevice));
    ParSet<T,I-1> p(pars...);
  }
};

template<class T>
struct ParSet<T,1>
{
  template<class U>
  ParSet(U u)
  {
    checkCudaErrors(cudaMemcpy(initCuFftStoreCallBack<T>::p_, &u,
          sizeof(void*), cudaMemcpyHostToDevice));
  }
};

struct cuFftStoreCallback
{
  template<class Exp>
    static void apply(cufftHandle plan)
    {

      static_assert(0==Exp::eParamNum,"Number of passed parameters "
          "have to be equal to  number of FloatUserGet identities!");
      cufftResult status = cufftXtSetCallback(plan,
          (void **)&storeCallbackVec[initCuFftStoreCallBack<Exp>::cbIdx_],
          CUFFT_CB_ST_COMPLEX,
          0);
      if (status == CUFFT_LICENSE_ERROR)
      {
        printf("License file was not found, out of date, or invalid.\n");
        exit(EXIT_FAILURE);
      }
      else
      {
        checkCudaErrors(status);
      }
    }
  template<class Exp,class...Ts>
    static void apply(cufftHandle plan,Ts... args)
    {
      static_assert(sizeof...(Ts)==Exp::eParamNum,"Number of passed parameters "
          "have to be equal to  number of FloatUserGet identities!");
      ParSet<Exp,sizeof...(args)> set(args...);
      // --- Associating the callback with the plan.
      cufftResult status = cufftXtSetCallback(plan,
          (void **)&storeCallbackVec[initCuFftStoreCallBack<Exp>::cbIdx_],
          CUFFT_CB_ST_COMPLEX,
          (void**)&initCuFftStoreCallBack<Exp>::p_);
      if (status == CUFFT_LICENSE_ERROR)
      {
        printf("License file was not found, out of date, or invalid.\n");
        exit(EXIT_FAILURE);
      }
      else
      {
        checkCudaErrors(status);
      }
    }
};

__device__ float2 operator/(float2 x, float a)
{
  return make_cuComplex(x.x/a,x.y/a);
}

__device__ float2 operator*(float2 x, float a)
{
  return make_cuComplex(x.x*a,x.y*a);
}

__device__ float2 operator-(float2 x, float2 y)
{
  return make_cuComplex(x.x-y.x,x.y-y.y);
}

__device__ float2 operator*(float2 x, float2 y)
{
  return make_cuComplex(x.x*y.x-x.y*y.y
      ,x.y*y.x+x.x*y.y);
}

__device__ float2 operator*=(float2& x, float y)
{
  x.x *= y;
  x.y *= y;
  return x;
}

__device__ float2 operator*=(float2& x, float2 y)
{
  return make_cuComplex(x.x*y.x-x.y*y.y
      ,x.y*y.x+x.x*y.y);
}

__host__ __device__ bool operator == (const float2& lhs, const float2& rhs)
{
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

__device__ cuComplex exp(cuComplex a)
{
  cuComplex r;
  float s,c;
  float e = expf(a.x);
  sincosf(a.y,&s,&c);
  r.x = c * e;
  r.y = s * e;
  return r;
}

__device__ float exp(float x)
{
  return expf(x);
}

__device__ float abs(float& x)
{
  return fabsf(x);
}

__device__ float abs(float2& x)
{
  return cuCabsf(x);
}
//
struct OpPlus
{
  template<typename T1, typename T2>
  __device__ T1 operator()(T1 x, T2 y)
  {
    return x+y;
  }
};

struct OpMinus
{
  template<typename T1, typename T2>
  __device__ T1 operator()(T1 x, T2 y)
  {
    return x-y;
  }
};

struct OpAbs
{
  template<typename T1>
  __device__ float operator()(T1 x)
  {
    return abs(x);
  }
};

struct OpMul
{
  template<typename T1, typename T2>
  __device__ T1 operator()(T1 x, T2 y)
  {
    return x*y;
  }
};

struct OpDiv
{
  template<typename T1, typename T2>
  __device__ T1 operator()(T1 x, T2 y)
  {
    return x/y;
  }
};

struct OpMulEq
{
  template<typename T1, typename T2>
  __device__ T1 operator()(T1& x, T2 y)
  {
    return x*=y;
  }
};


// Struct: CFloat
// Represents a complex float element of an array to which
// an FFT is applied
struct CFloat
{
  typedef float2 type;
  enum {eParamNum = 0};
  template<int I>
    __device__ float2& operator()(ParPos<I> l,float2& e,size_t of,void* p)
    {
      return e;
    }
};

struct GetIdentity
{
  template<class T>
    __device__ T get(T* v, size_t of)
    {
      return v[of];
    }
};

struct GetScalar
{
  template<class T>
    __device__ T get(T* v, size_t of)
    {
      return *v;
    }
};

template<class ExpT1, class ExpT2, class BinOp>
class BinExpr;

//
template<class GET = GetIdentity>
struct FloatUserGet
{
  typedef float type;
  enum {eParamNum = 1};
  template<int I>
    __device__ float operator()(ParPos<I> l,float2 e,size_t of,void* p)
    {
      float* ar =static_cast<float*>(*(static_cast<void**>(p)+I-1));
      return GET().get(ar,of);
    }
    template <class R>
      BinExpr<FloatUserGet<GET>,R,OpMul>
      operator*(R r)
      {
        return BinExpr<FloatUserGet<GET>,R,OpMul>();
      }
    template <class R>
      BinExpr<FloatUserGet<GET>,R,OpMinus>
      operator-(R r)
      {
        return BinExpr<FloatUserGet<GET>,R,OpMinus>();
      }
};
//
template<class GET = GetIdentity>
struct CFloatUserGet
{
  typedef float2 type;
  enum {eParamNum = 1};
  template<int I>
    __device__ float2 operator()(ParPos<I> l,float2 e,size_t of,void* p)
    {
      float2* ar =static_cast<float2*>(*(static_cast<void**>(p)+I-1));
      return GET().get(ar,of);
    }
    template <class R>
      BinExpr<CFloatUserGet<GET>,R,OpDiv>
      operator*(R r)
      {
        return BinExpr<CFloatUserGet<GET>,R,OpMul>();
      }
    template <class R>
      BinExpr<CFloatUserGet<GET>,R,OpMul>
      operator*(R r)
      {
        return BinExpr<CFloatUserGet<GET>,R,OpMul>();
      }
    template <class R>
      BinExpr<CFloatUserGet<GET>,R,OpPlus>
      operator+(R r)
      {
        return BinExpr<CFloatUserGet<GET>,R,OpPlus>();
      }
    template <class R>
      BinExpr<CFloatUserGet<GET>,R,OpMinus>
      operator-(R r)
      {
        return BinExpr<CFloatUserGet<GET>,R,OpMinus>();
      }
};
template<class ExpT1, class ExpT2, class BinOp>
class BinExpr
{
  public:
    typedef typename ExpT1::type type;
    enum {eParamNum = ExpT1::eParamNum+ExpT2::eParamNum};
    __device__ BinExpr() {}
    template<int I,typename... Ts>
    __device__ type operator ()(ParPos<I> l,Ts... args)
    {
      return _op(_expr1(ParPos<I>(),args...)
          ,_expr2(ParPos<I-ExpT1::eParamNum>(),args...) );
    }
    template <class R>
      BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpPlus>
      operator+(R r)
      {
        return BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpPlus>();
      }
    template <class R>
      BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpMinus>
      operator-(R r)
      {
        return BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpMinus>();
      }
    template <class R>
      BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpMul>
      operator*(R r)
      {
        return BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpMul>();
      }
    template <class R>
      BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpDiv>
      operator*(R r)
      {
        return BinExpr<BinExpr<ExpT1,ExpT2,BinOp>,R,OpDiv>();
      }
  private:
    ExpT1 _expr1;
    ExpT2 _expr2;
    BinOp _op;
};

template<class ExpT2>
BinExpr<CFloat,ExpT2,OpMulEq>
operator*=(CFloat e1, ExpT2 e2)
{
  return BinExpr<CFloat,ExpT2,OpMulEq>();
}

template<class ExpT2>
BinExpr<CFloat,ExpT2,OpPlus>
operator+(CFloat e1, ExpT2 e2)
{
  return BinExpr<CFloat,ExpT2,OpPlus>();
}
using CFloatUserScalar = CFloatUserGet<GetScalar>;
using FloatUserScalar = FloatUserGet<GetScalar>;
using CFloatUser = CFloatUserGet<GetIdentity>;
using FloatUser = FloatUserGet<GetIdentity>;

} // namespace cumgpu

#define CUFFT_STORE_CALLBACK_REG_CUSTOM( name, x ) \
  __device__ cufftCallbackStoreC name = \
  cufft_callback_store__<x>; \
  template<> \
  int initCuFftStoreCallBack<x>::cbIdx_ = 0;

#define CUFFT_STORE_CALLBACK_REG( name, x ) \
  typedef decltype(x) __T ## name ## __; \
  __device__ cufftCallbackStoreC name = \
  cufft_callback_store__<__T ## name ## __>; \
  template<> \
  int initCuFftStoreCallBack<__T ## name ## __>::cbIdx_ = 0; \
  template<> \
  void** initCuFftStoreCallBack<__T ## name ## __>::p_ = NULL;

#define CUFFT_STORE_CALLBACK_INIT_CUSTOM( name, x ) \
    checkCudaErrors(cudaMemcpyFromSymbol(&h_storeCallbackPtrTmp, \
          name, \
          sizeof(h_storeCallbackPtrTmp))); \
    storeCallbackVec.push_back(h_storeCallbackPtrTmp); \
    initCuFftStoreCallBack<x>::cbIdx_ = storeCallbackVec.size()- 1;

#define CUFFT_STORE_CALLBACK_INIT( name ) \
    checkCudaErrors(cudaMemcpyFromSymbol(&h_storeCallbackPtrTmp, \
          name, \
          sizeof(h_storeCallbackPtrTmp))); \
    storeCallbackVec.push_back(h_storeCallbackPtrTmp); \
    initCuFftStoreCallBack<__T ## name ## __>::cbIdx_ = \
      storeCallbackVec.size()- 1; \
    checkCudaErrors(cudaMalloc( \
          (void**)&initCuFftStoreCallBack<__T ## name ## __>::p_, \
          sizeof(void*)*(int(__T ## name ## __::eParamNum))));

#define REG_TYPE( n ) \
  __T ## n ## __


#endif /* end of include guard: CUFFT_CALLBACK_H_ZCIRVK4D */


// Copyright 2019 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "reverb/cc/conversions.h"

namespace deepmind {
namespace reverb {
namespace pybind {

void ImportNumpy() { import_array1(); }

struct PyDecrefDeleter {
  void operator()(PyObject *p) const { Py_DECREF(p); }
};
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject *o) { return Safe_PyObjectPtr(o); }

char const *NumpyTypeName(int numpy_type) {
  switch (numpy_type) {
#define TYPE_CASE(s) \
  case s:            \
    return #s

    TYPE_CASE(NPY_BOOL);
    TYPE_CASE(NPY_BYTE);
    TYPE_CASE(NPY_UBYTE);
    TYPE_CASE(NPY_SHORT);
    TYPE_CASE(NPY_USHORT);
    TYPE_CASE(NPY_INT);
    TYPE_CASE(NPY_UINT);
    TYPE_CASE(NPY_LONG);
    TYPE_CASE(NPY_ULONG);
    TYPE_CASE(NPY_LONGLONG);
    TYPE_CASE(NPY_ULONGLONG);
    TYPE_CASE(NPY_FLOAT);
    TYPE_CASE(NPY_DOUBLE);
    TYPE_CASE(NPY_LONGDOUBLE);
    TYPE_CASE(NPY_CFLOAT);
    TYPE_CASE(NPY_CDOUBLE);
    TYPE_CASE(NPY_CLONGDOUBLE);
    TYPE_CASE(NPY_OBJECT);
    TYPE_CASE(NPY_STRING);
    TYPE_CASE(NPY_UNICODE);
    TYPE_CASE(NPY_VOID);
    TYPE_CASE(NPY_DATETIME);
    TYPE_CASE(NPY_TIMEDELTA);
    TYPE_CASE(NPY_HALF);
    TYPE_CASE(NPY_NTYPES);
    TYPE_CASE(NPY_NOTYPE);
    TYPE_CASE(NPY_USERDEF);

    default:
      return "not a numpy type";
  }
}

tensorflow::Status PyObjectToString(PyObject *obj, const char **ptr,
                                    Py_ssize_t *len, PyObject **ptr_owner) {
  *ptr_owner = nullptr;
  if (PyBytes_Check(obj)) {
    char *buf;
    if (PyBytes_AsStringAndSize(obj, &buf, len) != 0) {
      return tensorflow::errors::Internal("Unable to get element as bytes.");
    }
    *ptr = buf;
  } else if (PyUnicode_Check(obj)) {
    *ptr = PyUnicode_AsUTF8AndSize(obj, len);
    if (*ptr == nullptr) {
      return tensorflow::errors::Internal("Unable to convert element to UTF-8");
    }
  } else {
    return tensorflow::errors::Internal("Unsupported object type ",
                                        obj->ob_type->tp_name);
  }

  return tensorflow::OkStatus();
}

// Iterate over the string array 'array', extract the ptr and len of each string
// element and call f(ptr, len).
template <typename F>
tensorflow::Status PyBytesArrayMap(PyArrayObject *array, F f) {
  auto iter = make_safe(PyArray_IterNew(reinterpret_cast<PyObject *>(array)));

  while (PyArray_ITER_NOTDONE(iter.get())) {
    auto item = make_safe(PyArray_GETITEM(
        array, static_cast<char *>(PyArray_ITER_DATA(iter.get()))));
    if (!item) {
      return tensorflow::errors::Internal(
          "Unable to get element from the feed - no item.");
    }
    Py_ssize_t len;
    const char *ptr;
    PyObject *ptr_owner = nullptr;
    TF_RETURN_IF_ERROR(PyObjectToString(item.get(), &ptr, &len, &ptr_owner));
    f(ptr, len);
    Py_XDECREF(ptr_owner);
    PyArray_ITER_NEXT(iter.get());
  }
  return tensorflow::OkStatus();
}

tensorflow::Status StringTensorToPyArray(const tensorflow::Tensor &tensor,
                                         PyArrayObject *dst) {
  DCHECK_EQ(tensor.dtype(), tensorflow::DT_STRING);

  auto iter = make_safe(PyArray_IterNew(reinterpret_cast<PyObject *>(dst)));

  const auto &flat_data = tensor.flat<tensorflow::tstring>().data();
  for (int i = 0; i < tensor.NumElements(); i++) {
    const auto &value = flat_data[i];
    auto py_string =
        make_safe(PyBytes_FromStringAndSize(value.c_str(), value.size()));
    if (py_string == nullptr) {
      return tensorflow::errors::Internal(
          "failed to create a python byte array when converting element #", i,
          " of a TF_STRING tensor to a numpy ndarray");
    }

    if (PyArray_SETITEM(dst, PyArray_ITER_DATA(iter.get()), py_string.get()) !=
        0) {
      return tensorflow::errors::Internal("Error settings element #", i,
                                          " in the numpy ndarray");
    }

    PyArray_ITER_NEXT(iter.get());
  }

  return tensorflow::OkStatus();
}

tensorflow::Status GetPyDescrFromDataType(tensorflow::DataType dtype,
                                          PyArray_Descr **out_descr) {
  switch (dtype) {
#define TF_TO_PY_ARRAY_TYPE_CASE(TF_DTYPE, PY_ARRAY_TYPE) \
  case TF_DTYPE:                                          \
    *out_descr = PyArray_DescrFromType(PY_ARRAY_TYPE);    \
    break;

    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_HALF, NPY_FLOAT16)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_FLOAT, NPY_FLOAT32)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_DOUBLE, NPY_FLOAT64)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_INT32, NPY_INT32)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_UINT8, NPY_UINT8)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_UINT16, NPY_UINT16)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_UINT32, NPY_UINT32)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_INT8, NPY_INT8)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_INT16, NPY_INT16)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_BOOL, NPY_BOOL)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_COMPLEX64, NPY_COMPLEX64)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_COMPLEX128, NPY_COMPLEX128)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_STRING, NPY_OBJECT)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_UINT64, NPY_UINT64)
    TF_TO_PY_ARRAY_TYPE_CASE(tensorflow::DT_INT64, NPY_INT64)

#undef TF_DTYPE_TO_PY_ARRAY_TYPE_CASE

    default:
      return tensorflow::errors::Internal(
          "Unsupported tf type: ", tensorflow::DataType_Name(dtype));
  }

  return tensorflow::OkStatus();
}

tensorflow::Status GetPyDescrFromTensor(const tensorflow::Tensor &tensor,
                                        PyArray_Descr **out_descr) {
  return GetPyDescrFromDataType(tensor.dtype(), out_descr);
}

tensorflow::Status GetTensorDtypeFromPyArray(
    PyArrayObject *array, tensorflow::DataType *out_tf_datatype) {
  int pyarray_type = PyArray_TYPE(array);
  switch (pyarray_type) {
#define NP_TO_TF_DTYPE_CASE(NP_DTYPE, TF_DTYPE) \
  case NP_DTYPE:                                \
    *out_tf_datatype = TF_DTYPE;                \
    break;

    NP_TO_TF_DTYPE_CASE(NPY_FLOAT16, tensorflow::DT_HALF)
    NP_TO_TF_DTYPE_CASE(NPY_FLOAT32, tensorflow::DT_FLOAT)
    NP_TO_TF_DTYPE_CASE(NPY_FLOAT64, tensorflow::DT_DOUBLE)

    NP_TO_TF_DTYPE_CASE(NPY_INT8, tensorflow::DT_INT8)
    NP_TO_TF_DTYPE_CASE(NPY_INT16, tensorflow::DT_INT16)
    NP_TO_TF_DTYPE_CASE(NPY_INT32, tensorflow::DT_INT32)
    NP_TO_TF_DTYPE_CASE(NPY_LONGLONG, tensorflow::DT_INT64)
    NP_TO_TF_DTYPE_CASE(NPY_INT64, tensorflow::DT_INT64)

    NP_TO_TF_DTYPE_CASE(NPY_UINT8, tensorflow::DT_UINT8)
    NP_TO_TF_DTYPE_CASE(NPY_UINT16, tensorflow::DT_UINT16)
    NP_TO_TF_DTYPE_CASE(NPY_UINT32, tensorflow::DT_UINT32)
    NP_TO_TF_DTYPE_CASE(NPY_ULONGLONG, tensorflow::DT_UINT64)
    NP_TO_TF_DTYPE_CASE(NPY_UINT64, tensorflow::DT_UINT64)

    NP_TO_TF_DTYPE_CASE(NPY_BOOL, tensorflow::DT_BOOL)

    NP_TO_TF_DTYPE_CASE(NPY_COMPLEX64, tensorflow::DT_COMPLEX64)
    NP_TO_TF_DTYPE_CASE(NPY_COMPLEX128, tensorflow::DT_COMPLEX128)

    NP_TO_TF_DTYPE_CASE(NPY_OBJECT, tensorflow::DT_STRING)
    NP_TO_TF_DTYPE_CASE(NPY_STRING, tensorflow::DT_STRING)
    NP_TO_TF_DTYPE_CASE(NPY_UNICODE, tensorflow::DT_STRING)

#undef NP_TO_TF_DTYPE_CASE

    default:

      return tensorflow::errors::Internal("Unsupported numpy type: ",
                                          NumpyTypeName(pyarray_type));
  }
  return tensorflow::OkStatus();
}

inline tensorflow::Status VerifyDtypeIsSupported(
    const tensorflow::DataType &dtype) {
  if (!tensorflow::DataTypeCanUseMemcpy(dtype) &&
      dtype != tensorflow::DT_STRING) {
    return tensorflow::errors::Unimplemented(
        "ndarrays that maps to tensors with dtype ",
        tensorflow::DataType_Name(dtype), " are not yet supported");
  }
  return tensorflow::OkStatus();
}

tensorflow::Status NdArrayToTensor(PyObject *ndarray,
                                   tensorflow::Tensor *out_tensor) {
  DCHECK(out_tensor != nullptr);
  auto array_safe = make_safe(PyArray_FromAny(
      /*op=*/ndarray,
      /*dtype=*/nullptr,
      /*min_depth=*/0,
      /*max_depth=*/0,
      /*requirements=*/NPY_ARRAY_CARRAY_RO,
      /*context=*/nullptr));
  if (!array_safe) {
    return tensorflow::errors::InvalidArgument(
        "Provided input could not be interpreted as an ndarray");
  }
  PyArrayObject *py_array = reinterpret_cast<PyArrayObject *>(array_safe.get());

  // Convert numpy dtype to TensorFlow dtype.
  tensorflow::DataType dtype;
  TF_RETURN_IF_ERROR(GetTensorDtypeFromPyArray(py_array, &dtype));
  TF_RETURN_IF_ERROR(VerifyDtypeIsSupported(dtype));

  absl::InlinedVector<tensorflow::int64, 4> dims(PyArray_NDIM(py_array));
  tensorflow::int64 nelems = 1;
  for (int i = 0; i < PyArray_NDIM(py_array); ++i) {
    dims[i] = PyArray_SHAPE(py_array)[i];
    nelems *= dims[i];
  }

  if (tensorflow::DataTypeCanUseMemcpy(dtype)) {
    *out_tensor = tensorflow::Tensor(dtype, tensorflow::TensorShape(dims));
    size_t size = PyArray_NBYTES(py_array);
    memcpy(out_tensor->data(), PyArray_DATA(py_array), size);
  } else if (dtype == tensorflow::DT_STRING) {
    *out_tensor = tensorflow::Tensor(dtype, tensorflow::TensorShape(dims));
    int i = 0;
    auto *out_t = out_tensor->flat<tensorflow::tstring>().data();
    TF_RETURN_IF_ERROR(
        PyBytesArrayMap(py_array, [out_t, &i](const char *ptr, Py_ssize_t len) {
          out_t[i++] = tensorflow::tstring(ptr, len);
        }));
  } else {
    return tensorflow::errors::Unimplemented("Unexpected dtype: ",
                                             tensorflow::DataTypeString(dtype));
  }

  return tensorflow::OkStatus();
}

tensorflow::Status TensorToNdArray(const tensorflow::Tensor &tensor,
                                   PyObject **out_ndarray) {
  TF_RETURN_IF_ERROR(VerifyDtypeIsSupported(tensor.dtype()));

  // Extract the numpy type and dimensions.
  PyArray_Descr *descr = nullptr;
  TF_RETURN_IF_ERROR(GetPyDescrFromTensor(tensor, &descr));

  absl::InlinedVector<npy_intp, 4> dims(tensor.dims());
  for (int i = 0; i < tensor.dims(); i++) {
    dims[i] = tensor.dim_size(i);
  }

  // Allocate an empty array of the desired shape and type.
  auto safe_out_ndarray =
      make_safe(PyArray_Empty(dims.size(), dims.data(), descr, 0));
  if (!safe_out_ndarray) {
    return tensorflow::errors::Internal("Could not allocate ndarray");
  }

  // Populate the ndarray with data from the tensor.
  PyArrayObject *py_array =
      reinterpret_cast<PyArrayObject *>(safe_out_ndarray.get());
  if (tensorflow::DataTypeCanUseMemcpy(tensor.dtype())) {
    memcpy(PyArray_DATA(py_array), tensor.data(), PyArray_NBYTES(py_array));
  } else if (tensor.dtype() == tensorflow::DT_STRING) {
    TF_RETURN_IF_ERROR(StringTensorToPyArray(tensor, py_array));
  } else {
    return tensorflow::errors::Unimplemented(
        "Unexpected tensor dtype: ",
        tensorflow::DataTypeString(tensor.dtype()));
  }

  *out_ndarray = safe_out_ndarray.release();
  return tensorflow::OkStatus();
}

}  // namespace pybind
}  // namespace reverb
}  // namespace deepmind

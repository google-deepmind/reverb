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

#include <memory>
#include <string>

#include "numpy/arrayobject.h"
#include "absl/container/inlined_vector.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "reverb/cc/checkpointing/interface.h"
#include "reverb/cc/client.h"
#include "reverb/cc/platform/checkpointing.h"
#include "reverb/cc/platform/server.h"
#include "reverb/cc/rate_limiter.h"
#include "reverb/cc/sampler.h"
#include "reverb/cc/selectors/fifo.h"
#include "reverb/cc/selectors/heap.h"
#include "reverb/cc/selectors/interface.h"
#include "reverb/cc/selectors/lifo.h"
#include "reverb/cc/selectors/prioritized.h"
#include "reverb/cc/selectors/uniform.h"
#include "reverb/cc/table.h"
#include "reverb/cc/table_extensions/interface.h"
#include "reverb/cc/writer.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace {

using ::tensorflow::error::Code;

struct PyDecrefDeleter {
  void operator()(PyObject *p) const { Py_DECREF(p); }
};
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject *o) { return Safe_PyObjectPtr(o); }

// Converts non OK statuses to Python exceptions and throws. Does nothing for
// OK statuses.
inline void MaybeRaiseFromStatus(const tensorflow::Status &status) {
  if (status.ok()) return;

  // TODO(b/152982733): Add tests that validates that casting behaviour is
  //   aligned with what tensorflow does.
  switch (status.code()) {
#define CODE_TO_PY_EXC(CODE, PY_EXC)                         \
  case CODE:                                                 \
    PyErr_SetString(PY_EXC, status.error_message().c_str()); \
    break;

    CODE_TO_PY_EXC(Code::INVALID_ARGUMENT, PyExc_ValueError)
    CODE_TO_PY_EXC(Code::RESOURCE_EXHAUSTED, PyExc_IndexError)
    CODE_TO_PY_EXC(Code::UNIMPLEMENTED, PyExc_NotImplementedError)
    CODE_TO_PY_EXC(Code::INTERNAL, PyExc_RuntimeError)

    // TODO(b/154927554): Map more status codes to Python exceptions.

#undef CODE_TO_PY_EXC

    default:
      PyErr_SetString(PyExc_RuntimeError, status.error_message().c_str());
  }

  throw pybind11::error_already_set();
}

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

void ImportNumpy() { import_array1(); }

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

  return tensorflow::Status::OK();
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
  return tensorflow::Status::OK();
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

  return tensorflow::Status::OK();
}

tensorflow::Status GetPyDescrFromTensor(const tensorflow::Tensor &tensor,
                                        PyArray_Descr **out_descr) {
  switch (tensor.dtype()) {
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
          "Unsupported tf type: ", tensorflow::DataType_Name(tensor.dtype()));
  }

  return tensorflow::Status::OK();
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

    case NPY_VOID:
      // TODO(b/154925774): Support struct and quantized types.
      return tensorflow::errors::Unimplemented(
          "Custom structs and quantized types are not supported");
    default:
      // TODO(b/154926401): Add support for bfloat16.
      // The bfloat16 type is defined in the internals of tf.
      if (pyarray_type == -1) {
        return tensorflow::errors::Unimplemented(
            "bfloat16 types are not yet supported");
      }

      return tensorflow::errors::Internal("Unsupported numpy type: ",
                                          NumpyTypeName(pyarray_type));
  }
  return tensorflow::Status::OK();
}

inline tensorflow::Status VerifyDtypeIsSupported(
    const tensorflow::DataType &dtype) {
  if (!tensorflow::DataTypeCanUseMemcpy(dtype) &&
      dtype != tensorflow::DT_STRING) {
    return tensorflow::errors::Unimplemented(
        "ndarrays that maps to tensors with dtype ",
        tensorflow::DataType_Name(dtype), " are not yet supported");
  }
  return tensorflow::Status::OK();
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

  return tensorflow::Status::OK();
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
  return tensorflow::Status::OK();
}

}  // namespace

namespace pybind11 {
namespace detail {

// Convert between absl::optional and python.
//
// pybind11 supports std::optional, and absl::optional is meant to be a
// drop-in replacement for std::optional, so we can just use the built in
// implementation.
//
// If we start getting errors due to this being defined in multiple places that
// likely means that pybind11 has included the cast itself and we can remove
// this implementation.
#ifndef ABSL_USES_STD_OPTIONAL
template <typename T>
struct type_caster<absl::optional<T>>
    : public optional_caster<absl::optional<T>> {};

template <>
struct type_caster<absl::nullopt_t> : public void_caster<absl::nullopt_t> {};
#endif

template <>
struct type_caster<tensorflow::Tensor> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::Tensor, _("tensorflow::Tensor"));

  bool load(handle handle, bool) {
    tensorflow::Status status = NdArrayToTensor(handle.ptr(), &value);

    if (!status.ok()) {
      std::string message = status.ToString();
      REVERB_LOG(REVERB_ERROR)
          << "Tensor can't be extracted from the source represented as "
             "ndarray: "
          << message;
      // When a conversion fails, PyErr is set. Returning from `load` with PyErr
      // set results in crashes so we clear the error here to make the Python
      // error slightly more readable.
      PyErr_Clear();
      return false;
    }
    return true;
  }

  static handle cast(const tensorflow::Tensor &src, return_value_policy,
                     handle) {
    PyObject *ret;
    tensorflow::Status status = TensorToNdArray(src, &ret);
    if (!status.ok()) {
      std::string message = status.ToString();
      PyErr_SetString(PyExc_ValueError, message.data());
      return nullptr;
    }
    return ret;
  }
};

// Raise an exception if a given status is not OK, otherwise return None.
template <>
struct type_caster<tensorflow::Status> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::Status, _("Status"));
  static handle cast(tensorflow::Status status, return_value_policy, handle) {
    MaybeRaiseFromStatus(status);
    return none().inc_ref();
  }
};

}  // namespace detail
}  // namespace pybind11

namespace deepmind {
namespace reverb {
namespace {

namespace py = pybind11;

PYBIND11_MODULE(libpybind, m) {
  // Initialization code to use numpy types in the type casters.
  ImportNumpy();

  py::class_<ItemSelectorInterface, std::shared_ptr<ItemSelectorInterface>>
      unused_key_distribution_interface(m, "ItemSelectorInterface");

  py::class_<PrioritizedSelector, ItemSelectorInterface,
             std::shared_ptr<PrioritizedSelector>>(m, "PrioritizedSelector")
      .def(py::init<double>(), py::arg("priority_exponent"));

  py::class_<FifoSelector, ItemSelectorInterface,
             std::shared_ptr<FifoSelector>>(m, "FifoSelector")
      .def(py::init());

  py::class_<LifoSelector, ItemSelectorInterface,
             std::shared_ptr<LifoSelector>>(m, "LifoSelector")
      .def(py::init());

  py::class_<UniformSelector, ItemSelectorInterface,
             std::shared_ptr<UniformSelector>>(m, "UniformSelector")
      .def(py::init());

  py::class_<HeapSelector, ItemSelectorInterface,
             std::shared_ptr<HeapSelector>>(m, "HeapSelector")
      .def(py::init<bool>(), py::arg("min_heap"));

  py::class_<TableExtensionInterface, std::shared_ptr<TableExtensionInterface>>
      unused_priority_table_extension_interface(m, "TableExtensionInterface");

  py::class_<RateLimiter, std::shared_ptr<RateLimiter>>(m, "RateLimiter")
      .def(py::init<double, int, double, double>(),
           py::arg("samples_per_insert"), py::arg("min_size_to_sample"),
           py::arg("min_diff"), py::arg("max_diff"));

  py::class_<Table, std::shared_ptr<Table>>(m, "Table")
      .def(py::init(
               [](const std::string &name,
                  const std::shared_ptr<ItemSelectorInterface> &sampler,
                  const std::shared_ptr<ItemSelectorInterface> &remover,
                  int max_size, int max_times_sampled,
                  const std::shared_ptr<RateLimiter> &rate_limiter,
                  const std::vector<std::shared_ptr<TableExtensionInterface>>
                      &extensions,
                  const absl::optional<std::string> &serialized_signature =
                      absl::nullopt) -> Table * {
                 absl::optional<tensorflow::StructuredValue> signature =
                     absl::nullopt;
                 if (serialized_signature) {
                   signature.emplace();
                   if (!signature->ParseFromString(*serialized_signature)) {
                     MaybeRaiseFromStatus(tensorflow::errors::InvalidArgument(
                         "Unable to deserialize StructuredValue from "
                         "serialized proto bytes: '",
                         *serialized_signature, "'"));
                     return nullptr;
                   }
                 }
                 return new Table(name, sampler, remover, max_size,
                                  max_times_sampled, rate_limiter, extensions,
                                  std::move(signature));
               }),
           py::arg("name"), py::arg("sampler"), py::arg("remover"),
           py::arg("max_size"), py::arg("max_times_sampled"),
           py::arg("rate_limiter"), py::arg("extensions"), py::arg("signature"))
      .def("name", &Table::name)
      .def("can_sample", &Table::CanSample,
           py::call_guard<py::gil_scoped_release>())
      .def("can_insert", &Table::CanInsert,
           py::call_guard<py::gil_scoped_release>());

  py::class_<Writer>(m, "Writer")
      .def("Append", &Writer::Append, py::call_guard<py::gil_scoped_release>())
      .def("CreateItem", &Writer::CreateItem,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "Flush",
          [](Writer *writer) { MaybeRaiseFromStatus(writer->Flush()); },
          py::call_guard<py::gil_scoped_release>())
      .def("Close", &Writer::Close, py::call_guard<py::gil_scoped_release>());

  py::class_<Sampler>(m, "Sampler")
      .def(
          "GetNextTimestep",
          [](Sampler *sampler) {
            std::vector<tensorflow::Tensor> sample;
            bool end_of_sequence;
            MaybeRaiseFromStatus(
                sampler->GetNextTimestep(&sample, &end_of_sequence));
            return std::make_pair(std::move(sample), end_of_sequence);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("Close", &Sampler::Close, py::call_guard<py::gil_scoped_release>());

  py::class_<Client>(m, "Client")
      .def(py::init<std::string>(), py::arg("server_name"))
      .def(
          "NewWriter",
          [](Client *client, int chunk_length, int max_timesteps,
             bool delta_encoded, absl::optional<int> max_in_flight_items) {
            std::unique_ptr<Writer> writer;
            MaybeRaiseFromStatus(
                client->NewWriter(chunk_length, max_timesteps, delta_encoded,
                                  std::move(max_in_flight_items), &writer));
            return writer;
          },
          py::call_guard<py::gil_scoped_release>(), py::arg("chunk_length"),
          py::arg("max_timesteps"), py::arg("delta_encoded") = false,
          py::arg("max_in_flight_items") = absl::nullopt)
      .def(
          "NewSampler",
          [](Client *client, const std::string &table, int64_t max_samples,
             size_t buffer_size, int64_t validation_timeout_ms) {
            std::unique_ptr<Sampler> sampler;
            Sampler::Options options;
            options.max_samples = max_samples;
            options.max_in_flight_samples_per_worker = buffer_size;
            absl::Duration validation_timeout =
                (validation_timeout_ms < 0)
                    ? absl::InfiniteDuration()
                    : absl::Milliseconds(validation_timeout_ms);
            MaybeRaiseFromStatus(client->NewSampler(
                table, options, validation_timeout, &sampler));
            return sampler;
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "MutatePriorities",
          [](Client *client, const std::string &table,
             const std::vector<std::pair<uint64_t, double>> &updates,
             const std::vector<uint64_t> &deletes) {
            std::vector<KeyWithPriority> update_protos;
            for (const auto &update : updates) {
              update_protos.emplace_back();
              update_protos.back().set_key(update.first);
              update_protos.back().set_priority(update.second);
            }
            return client->MutatePriorities(table, update_protos, deletes);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("Reset", &Client::Reset, py::call_guard<py::gil_scoped_release>())
      .def("ServerInfo",
           [](Client *client, int timeout_sec) {
             // Wait indefinetely for server to startup when timeout not
             // provided.
             auto timeout = timeout_sec > 0 ? absl::Seconds(timeout_sec)
                                            : absl::InfiniteDuration();

             struct Client::ServerInfo info;

             // Release the GIL only when waiting for the call to complete. If
             // the GIL is not held when `MaybeRaiseFromStatus` is called it can
             // result in segfaults as the Python exception is populated with
             // details from the status.
             tensorflow::Status status;
             {
               py::gil_scoped_release g;
               status = client->ServerInfo(timeout, &info);
             }
             MaybeRaiseFromStatus(status);

             // Return a serialized ServerInfo proto bytes string.
             std::vector<py::bytes> serialized_table_info;
             serialized_table_info.reserve(info.table_info.size());
             for (const auto &table_info : info.table_info) {
               serialized_table_info.push_back(
                   py::bytes(table_info.SerializeAsString()));
             }
             return serialized_table_info;
           })
      .def("Checkpoint", [](Client *client) {
        std::string path;
        tensorflow::Status status;
        {
          py::gil_scoped_release g;
          status = client->Checkpoint(&path);
        }
        MaybeRaiseFromStatus(status);
        return path;
      });

  py::class_<CheckpointerInterface, std::shared_ptr<CheckpointerInterface>>
      unused_checkpointer_interface(m, "CheckpointerInterface");

  m.def(
      "create_default_checkpointer",
      [](const std::string &name, const std::string &group = "") {
        auto checkpointer = CreateDefaultCheckpointer(name, group);
        return std::shared_ptr<CheckpointerInterface>(checkpointer.release());
      },
      py::call_guard<py::gil_scoped_release>());

  py::class_<Server, std::shared_ptr<Server>>(m, "Server")
      .def(
          py::init([](std::vector<std::shared_ptr<Table>> priority_tables,
                      int port,
                      std::shared_ptr<CheckpointerInterface> checkpointer =
                          nullptr) {
            std::unique_ptr<Server> server;
            MaybeRaiseFromStatus(StartServer(std::move(priority_tables), port,
                                             std::move(checkpointer), &server));
            return server.release();
          }),
          py::arg("priority_tables"), py::arg("port"),
          py::arg("checkpointer") = nullptr)
      .def("Stop", &Server::Stop, py::call_guard<py::gil_scoped_release>())
      .def("Wait", &Server::Wait, py::call_guard<py::gil_scoped_release>())
      .def("InProcessClient", &Server::InProcessClient,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace
}  // namespace reverb
}  // namespace deepmind

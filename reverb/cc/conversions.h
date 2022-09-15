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

#ifndef REVERB_CC_CONVERSIONS_H_
#define REVERB_CC_CONVERSIONS_H_

#include "numpy/arrayobject.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"

namespace deepmind {
namespace reverb {
namespace pybind {

// One MUST initialize Numpy, e.g. within the Pybind11 module definition before
// calling C Numpy functions.
// See https://pythonextensionpatterns.readthedocs.io/en/latest/cpp_and_numpy.html
void ImportNumpy();

tensorflow::Status TensorToNdArray(const tensorflow::Tensor &tensor,
                                   PyObject **out_ndarray);

tensorflow::Status NdArrayToTensor(PyObject *ndarray,
                                   tensorflow::Tensor *out_tensor);

tensorflow::Status GetPyDescrFromDataType(tensorflow::DataType dtype,
                                          PyArray_Descr **out_descr);

}  // namespace pybind
}  // namespace reverb
}  // namespace deepmind

#endif  // REVERB_CC_CONVERSIONS_H_

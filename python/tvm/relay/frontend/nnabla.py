# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" NNabla: Neural Network Libraries Frontend for Relay """
import numpy as np
import collections
import zipfile
import shutil
import tempfile
import logging
import os
import attr  
import sys 
logging.basicConfig(level=logging.CRITICAL)
import nnabla as nn
from nnabla.utils import nnabla_pb2
from nnabla.utils.converter.nnabla import NnpImporter
import google.protobuf.text_format as text_format 

import tvm 
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis 
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ..expr_functor import ExprFunctor 

from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_shape, infer_channels
from .common import infer_type, get_name
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated

from . import qnn_nnabla

__all__ = ['from_nnabla']

# #############################################################################
# Helper functions
# ----------------

def load_nnp(nnp_path):
    """ Load nnp file and create a NnpImporter object with protobuf parameters which
        will be later used to convert into the Relay IR.
        This function only is usable to parse the NNabla graph into the Relay IR.
        To execute the Nnp file with NNabla use the NnpLoader tool instead.

        Parameters
        ----------
        nnp_path : str
            Path to the .nnp file from NNabla.

        Returns
        -------
        net : nnabla.utils.converter.nnabla.importer.NnpImporter
            Imported nnp file 
        """

    """  This function only is usable to parse the NNabla graph into the Relay IR. """
    net = NnpImporter(nnp_path, expad_network=False, executor_index=True)
    
    return net.execute()

def default_layout(dims, op_name):
    """
    A helper function to get default layout 
    """
    if dims == 1:
        return 'NCW'
    elif dims == 2:
        return 'NCHW'
    elif dims == 3:
        pass
        # return 'NCDHW'

    msg = "Only 1d and 2d layouts are currently supported"
    raise tvm.error.OpAttributeInvalid(msg.format(op_name))

def  dimension_picker(prefix, suffix=''):
    """ Check that dimensions are supported """
    # TODO: Check variables names
    def _impl(attrs):
        kernel = attrs['pool_size']
        if len(kernel) == 1:
            return prefix + '1d' + suffix 
        if len(kernel) == 2:
            return prefix + '2d' + suffix 
        if len(kernel) == 3:
            return prefix + '3d' + suffix 
        msg = 'Only 1D and 2D kernels are supported for operator {}.'
        op_name = prefix + '1d/2d'
        raise tvm.error.OpAttributeInvalid(msg.format(op_name))
    
    return _impl

def dimension_constraint():
    """ A helper function to restric dimensions """
    def _dim_check(attrs):
        if len(attrs['pool_size']) in [1, 2, 3]:
            return True
        return False 
    
    return _dim_check, "Only 1d, 2d, 3d kernel supported."

def replace_negative_size_with_batch_size(shape, batch_size):
    """Replace all dimensions with negative values to batch size"""
    sl = []
    for d in shape.dim:
        if d < 0:
            # Negative size means batch size
            sl.append(batch_size)
        else:
            sl.append(d)
    out_shape = nnabla_pb2.Shape()
    out_shape.dim.extend(sl)
    return list(out_shape.dim)

def infer_nnabla_shape(shape):
    tmp_value = 1
    shape = list(shape)
    for s in shape:
        tmp_value *= s 
    if len(shape) == 4 and max(shape) == tmp_value:
        return [max(shape)]
    else:
        return shape

class nnabla_input():
    """ Dual purpose list or dictionary access object. """
    
    def __init__(self):
        self.input_keys = []
        self.input_dict = {}

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.input_dict[self.input_keys[item]]
        if isinstance(item, str):
            if item not in self.input_keys:
                return None 
            return self.input_dict[item]
        if isinstance(item, slice):
            keys = self.input_keys[item]
            return [self.input_dict[key] for key in keys]
        
        raise ValueError("Only integer, string, and slice accesses allowed")
    
    def __setitem__(self, item, value):
        if isinstance(item, int):
            self.input_dict[self.input_keys[item]] = value 
        elif isinstance(item, str):
            self.input_keys.append(item)
            self.input_dict[item] = value 
        else:
            raise ValueError("Only integer, string, and slice accesses allowed")
    
    def keys(self):
        return self.input_keys

    def replace(self, old_item, item, value):
        """ Method defined to replace full-precision weights with quantized values """
        if isinstance(item, int):
            # Remove old value
            self.input_dict.pop(self.input_keys[old_item])
            self.input_keys.remove(old_item)

            # Insert new value
            self.input_dict[self.input_keys[item]] = value
        elif isinstance(item, str):
            # Remove old value
            self.input_dict.pop(old_item)
            self.input_keys.remove(old_item)

            # Insert new value
            self.input_keys.append(item)
            self.input_dict[item] = value

    def __len__(self):
        return len(self.input_keys)

    def __iter__(self):
        self.n = 0
        return self 
    
    def __next__(self):
        if self.n < len(self.input_keys):
            output = self.input_dict[self.input_keys[self.n]]
            self.n += 1
            return output 
        
        raise StopIteration

def get_tensor_type(name, type_dict):
    # TODO: Grab tensor type and provide proper type to convert to numpy array
    # This function should return the numpy type (e.g.: np.float32) to convert
    # NNabla tensor into numpy array. Method should be called inside _parse_array
    # function defined inside the Exporter object.
    return None

def check_data_format():
    # TODO: Define _check_data_format function
    #       This function should check the layout of each layer,
    #       it should return "NCWH" (channel_first) or "NWHC" (channel_first),
    #       this will allow to use Relays Data Layout transformation for
    #       Optimization.
    return None

# #############################################################################
# Operator definition
# -------------------
# 
# Nnabla operators are grouped in different converters 
# (e.g.: Activations in _convert_activations), specific functions have their
# own converter (e.g.: 2D Convolution as _convert_convolution). 
# Quantizer converters are defined in qnn_nnabla.py

def _none():
    def _impl(inputs, func, shapes):
        print(func.name)
        return None 
    return _impl 

def _convert_reshape():
    def _impl(inputs, func, shapes):
        if hasattr(func.reshape_param, 'shape'):
            shape = tuple(func.reshape_param.shape.dim)
            return _op.reshape(inputs[0], shape)
        else:
            raise NotImplementedError("Dinamic input case not yet supported")
            # return _op.reshape(inputs[0], inputs[1])
    return _impl

def _convert_concat():
    def _impl(inputs, func, shapes):
        # TODO: check data layout
        assert len(inputs) == 2
        axis = func.concatenate_param.axis
        return _op.concatenate(inputs[:], axis=axis)
    return _impl

def _convert_activation():
    def _impl(inputs, func, shapes):
        act_type = func.type
        data = inputs[0]

        if act_type == 'Softmax':
            return _op.nn.softmax(data, axes=1)
        if act_type == 'ReLU':
            return _op.nn.relu(data)
        if act_type == 'ReLU6':
            return _op.clip(data, a_min=0., a_max=6.)
        if act_type == 'Tanh':
            return _op.tanh(data)
        if act_type == 'Sigmoid':
            return _op.sigmoid(data)
        
        raise tvm.error.OpNotImplemented(
            'Activation Operator {} is not yet supported \
             with frontend NNabla'.format(act_type))

    return _impl 

def _convert_convolution():
    def _impl(inputs, func, shapes):
        # TODO: Map layouts. For that, include the shape dict from Exporter in order to get 
        # channel size and kernel size. data layout can be inferred with the shape
        # TODO: Check for all possible input combinations
        # for stride, pads, dilation, groups, channels, kernel_size 
        # TODO: data and kernel layouts should be infered with check_data_format method
        data_layout = "NCHW"
        kernel_layout = "OIHW"
        
        # Extract information from nnabla node found in convolution_param
        _stride = tuple(func.convolution_param.stride.dim)
        _pad_w = func.convolution_param.pad.dim[0]
        _pad_h = func.convolution_param.pad.dim[1]
        _pad = (_pad_w, _pad_h, _pad_w, _pad_h)
        _dilation = tuple(func.convolution_param.dilation.dim)
        _group = func.convolution_param.group
        _output_channels = shapes[func.input[1]][0]
        _kernel_shape = tuple(shapes[func.input[1]][2:])

        conv_out = _op.nn.conv2d(inputs[0],
                                 inputs[1],
                                 strides=_stride,
                                 padding=_pad,
                                 dilation=_dilation,
                                 groups=_group,
                                 channels= _output_channels,
                                 kernel_size= _kernel_shape,
                                 data_layout=data_layout,
                                 kernel_layout=kernel_layout,
                                 out_layout="",
                                 out_dtype="")
        
        use_bias = len(inputs) == 3

        if isinstance(shapes[-1], float):
            conv_out = _op.multiply(conv_out, _expr.const(shapes[-1], dtype="float32"))

        if use_bias:
            return _op.nn.bias_add(conv_out, inputs[2])
        else:
            return conv_out 
    
    return _impl


def _convert_gemm():
    def _impl(inputs, func, shapes):
        # Equivalent Op to GEMM in ONNX
        # Y = alpha * A * B + beta * C(If exists)

        # TODO: Infer values from NNabla Parameters
        alpha = float(1.0)
        beta = float(1.0)
        transA = 0
        transB = 0

        # get number of channels 
        channels = infer_channels(inputs[1], not transB)
        if transA:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not transB: 
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        inputs[0] = _op.nn.batch_flatten(inputs[0])
        
        if alpha != 1.0:
            inputs[0] *= _expr.const(alpha)
        out = _op.nn.dense(inputs[0], inputs[1], units=channels)
        
        use_bias = len(inputs) == 3
        if use_bias or (beta != 0.0):
            return _op.nn.bias_add(out, _expr.const(beta) * inputs[2])
        else:
            return out 

    return _impl

def _convert_advanced_activation():
    def _impl(inputs, func, shapes):
        # TODO: Create activation operators with clip values
        return None 
    
    return _impl

def _convert_pooling():
    def _impl(inputs, func, shapes):
        # Get data_layout with check_data_format
        pool_type = func.type
        data_layout = default_layout(len(shapes[0]) - 2, pool_type)
        if pool_type in ['GlobalMaxPooling','GlobalAveragePooling']:
            raise tvm.error.OpNotImplemented(
                'Function {} has experimental support in Nnabla frontend, \
                please do not use it'.format(pool_type))
        if pool_type == 'AveragePooling':
            # ignore_border is not considered
            attrs = {'pool_size': tuple(func.average_pooling_param.kernel.dim),
                     'strides': tuple(func.average_pooling_param.stride.dim),
                     'padding': tuple(func.average_pooling_param.pad.dim),
                     'count_include_pad': func.average_pooling_param.including_pad,
                     'layout': data_layout}
            return AttrCvt(op_name=dimension_picker("avg_pool"),
                           custom_check=dimension_constraint())(inputs, attrs)
            # return _op.nn.avg_pool2d(inputs[0], **attrs)
        if pool_type == 'MaxPooling':
            attrs = {'pool_size': tuple(func.max_pooling_param.kernel.dim),
                     'strides': tuple(func.max_pooling_param.stride.dim),
                     'padding': tuple(func.max_pooling_param.pad.dim),
                     'layout': data_layout}

            return AttrCvt(op_name=dimension_picker("max_pool"),
                           custom_check=dimension_constraint())(inputs, attrs)
            # return _op.nn.max_pool2d(inputs[0], **attrs)

        raise tvm.error.OpNotImplemented(
            'Pooling Operator {} is not yet supported \
             with frontend NNabla'.format(pool_type))

    return _impl

def _convert_batchnorm():
    def _impl(inputs, func, shapes):
        # TODO: Check data layout
        # TODO: Decompose inputs in a proper way with scale and center booleans
        # gamma, beta, moving_mean and moving_var should be extracted from the inputs

        # Ignore momentum/decay_rate
        # By default NNabla includes scale and bias term, otherwise no_scale and no_bias 
        # should be included in the parameters 
        attrs = {'beta': inputs[1],
                  'gamma': inputs[2],
                  'moving_mean': inputs[3],
                  'moving_var': inputs[4],
                  'axis': func.batch_normalization_param.axes[0],
                  'epsilon': func.batch_normalization_param.eps,
                  'center': True,
                  'scale': True}
  
        result, moving_mean, moving_var = _op.nn.batch_norm(inputs[0], **attrs)

        return result

    return _impl 

def _convert_elemwise():
    """ Element-wise operators between two arrays"""

    def _impl(inputs, func, shapes):
        op_name = func.type
        assert len(inputs) == 2, "Math operator {} take 2 inputs, {} given".format(op_name, len(inputs))
        if op_name in ['Add2', 'Mul2', 'Div2', 'Sub2', 'Pow2', 'Minimum2','Maximum2']:
            # Operations with numpy-style broadcasting
            op_map = {'Add2': _op.add,
                      'Mul2': _op.multiply,
                      'Div2': _op.divide,
                      'Sub2': _op.subtract,
                      'Pow2': _op.power,
                      'Minimum2': _op.minimum,
                      'Maximum2': _op.maximum}
            result = op_map[op_name](inputs[0], inputs[1])
        elif op_name in ['Less', 'Greater', 'Equal']:
            # Broadcasted elementwise operators
            op_map = {'Less': _op.less,
                      'Greater': _op.greater,
                      'Equal': _op.equal}
            result = op_map[op_name](inputs[0], inputs[1])
        elif op_name in ['LogicalAnd', 'LogicalNot', 'LogicalOr', 'LogicalXor']:
            # Logical operators with numpy-style broadcasting
            op_map = {'LogicalAnd': _op.logical_and,
                      'LogicalNot': _op.logical_not,
                      'LogicalOr': _op.logical_or,
                      'LogicalXor': _op.logical_xor}
            if op_name == 'LogicalNot':
                result = op_map[op_name](inputs[0])
            else:
                result = op_map[op_name](inputs[0], inputs[1])
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not yet implemented with frontend Nnabla'.format(op_name))
        
        return result

    return _impl

def _convert_scalar_elemwise():
    """ Element-wise operators that involve only one input array or one input
        array with a scalar value."""
    
    def _impl(inputs, func, shapes):
        op_name = func.type
        if op_name in ['AddScalar', 'MinimumScalar', 'MaximumScalar', 'PowScalar',
                       'MulScalar']:
            op_map = {'AddScalar': [_op.add, func.add_scalar_param.val],
                      'PowScalar': [_op.power, func.pow_scalar_param.val],
                      'MinimumScalar': [_op.minimum, func.minimum_scalar_param.val],
                      'MaximumScalar': [_op.maximum, func.maximum_scalar_param.val],
                      'MulScalar': [_op.multiply, func.mul_scalar_param.val]}

            value = op_map[op_name][1]
            result = op_map[op_name][0](inputs[0], _expr.const(value))
        elif op_name in ['RPowScalar']:
            op_map = {'RPowScalar': [_op.power, func.r_pow_scalar_param.val]}

            value = op_map[op_name][1]
            result = op_map[op_name][0](_expr.const(value), inputs[0])
        elif op_name == 'Round':
            result = _op.round(inputs[0])
        elif op_name == 'Log':
            result = _op.log(inputs[0])
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not yet implemented with frontend Nnabla'.format(op_name))

        return result
    return _impl

def _convert_broadcast():
    def _impl(inputs, func, shapes):
        new_shape = tuple(func.broadcast_param.shape.dim)

        return _op.broadcast_to(inputs[0], new_shape)
    return _impl

def _convert_flatten():
    def _impl(inputs, func, shapes):
        # TODO: check data layout
        return _op.nn.batch_flatten(inputs[0])

    return _impl

def _convert_clip():
    def _impl(inputs, func, shapes):
        # TODO: Complete with clip operator
        return None
    return _impl

def _convert_affine():
    def _impl(inputs, func, shapes):
        """
         Affine layer, also called as fully connected layer
         is decomposed in 3 steps (similar to ONNX converter):
         - Reshape inputs
         - GEMM
         - Reshape
        """ 
        # base_axis:  Base axis of Affine operation.
        base_axis = func.affine_param.base_axis
        x_shape = shapes[0]
        w_shape = shapes[1]
        y_shape = list(shapes[func.output[0]])
        x_shape_dims = [np.prod(x_shape[:base_axis]),
                        np.prod(x_shape[base_axis:])]
        gemm_output_shape = [np.prod(x_shape[:base_axis]),
                             np.prod(w_shape[1:])]
        
        # Reshape Inputs                    
        if x_shape_dims != x_shape:
            inputs[0] = _op.reshape(inputs[0], x_shape_dims)
        
        # GEMM
        if gemm_output_shape == y_shape:
            return _convert_gemm()(inputs, func, shapes[:-1])
        else:
            raise tvm.error.OpAttributeInvalid(
                'Operator {} does not currently support different \
                output shapes'.format(func.type))

    return _impl

# #############################################################################
# Converter map for NNabla 
# ------------------------
# 
# NNabla operators linked to the Relay converter
# 
_convert_map = {
    'SoftMax'                  : _convert_activation(),
    'ReLU'                     : _convert_activation(),
    'ReLU6'                    : _convert_activation(),
    'LeakyReLU'                : _convert_activation(),
    'PReLU'                    : _convert_activation(),
    'ELU'                      : _convert_activation(),
    'SELU'                     : _convert_activation(),
    'Sigmoid'                  : _convert_activation(),
    'Tanh'                     : _convert_activation(),

    'AveragePooling'           : _convert_pooling(),
    'MaxPooling'               : _convert_pooling(),
    'GlobalAveragePooling'     : _convert_pooling(),
    'GlobalMaxPooling'         : _convert_pooling(),
    'Convolution'              : _convert_convolution(),
    'Conv2DTranspose'          : _none(),
    'DepthwiseConv2D'          : _none(),

    'Add2'                     : _convert_elemwise(),
    'Mul2'                     : _convert_elemwise(),
    'Div2'                     : _convert_elemwise(),
    'Pow2'                     : _convert_elemwise(),
    'Sub2'                     : _convert_elemwise(),
    'Maximum2'                 : _convert_elemwise(),
    'Minimum2'                 : _convert_elemwise(),
    'Less'                     : _convert_elemwise(),
    'Greater'                  : _convert_elemwise(),
    'Equal'                    : _convert_elemwise(),
    'LogicalAnd'               : _convert_elemwise(),
    'LogicalNot'               : _convert_elemwise(),
    'LogicalOr'                : _convert_elemwise(),
    'LogicalXor'               : _convert_elemwise(),
    'MaximumScalar'            : _convert_scalar_elemwise(),
    'MinimumScalar'            : _convert_scalar_elemwise(),
    'PowScalar'                : _convert_scalar_elemwise(),
    'RPowScalar'               : _convert_scalar_elemwise(),
    'AddScalar'                : _convert_scalar_elemwise(),
    'MulScalar'                : _convert_scalar_elemwise(),
    'Log'                      : _convert_scalar_elemwise(),
    'Round'                    : _convert_scalar_elemwise(),
    'Broadcast'                : _convert_broadcast(),
    'Clip'                     : _convert_clip(),


    'Flatten'                  : _convert_flatten(),
    'Reshape'                  : _convert_reshape(),
    'Concatenate'              : _convert_concat(),
    'BatchNormalization'       : _convert_batchnorm(),
    'Affine'                   : _convert_affine(),

    'FixedPointQuantize'       : _none(),
    'Pow2Quantize'             : _none(),   
}

# #############################################################################
# NNabla converter definiton
# --------------------------

def get_converter(op):
    """ Convert NNabla operators to Relay Converter """
    return _convert_map[op]

class NNablaGraph(object):
    """ NNabla Graph data extracted from .nnp file
    Object initializes the network saved in the loaded .nnp file and
    extracts key information to convert the operators into Relay IR.
    Follows the method in which NNabla exports the graph to ONNX GraphProto.
    """
    def __init__(self, nnp, shape, dtype, batch_size=1):
        # NNabla related variables
        self._nnp = nnp.protobuf        # nnabla graph as protobuf object
        self._batch_size = batch_size   # executor batch_size
        self._net = None                # network_name
        self._executor = None
        self._parameters = {}
        self._var_dict = {}
        self.initializer = {}
        self.inputs = {}
        self.outputs = {}
        self.nodes = {}

    def _set_network(self):
        if len(self._nnp.executor) != 1:
            raise ValueError(
                "NNP with only a single executor is supported!")
        exe = self._nnp.executor[0]

        net = None 
        for n in self._nnp.network:
            if n.name == exe.network_name:
                net = n 
        if net is None:
            raise ValueError(
                "Executor network [{}] is not found in the NNP file.".format(exe.network_name))
        self._net = net 
        self._executor = exe 
        return net
    
    def _set_shape_all(self):
        assert isinstance(self._batch_size, int)
        bs = self._batch_size
        if bs < 0:
            bs = self._net._batch_size
        self._batch_size = bs
        # store all variable shape info to use later 
        for v in self._net.variable:
            self._var_dict[v.name] = replace_negative_size_with_batch_size(
                v.shape, bs)
        
        for p in self._nnp.parameter:
            self._parameters[p.variable_name] = p
    
    def _set_variables(self):
        exe = self._executor
        for param in self._nnp.parameter:
            if param.variable_name in self._var_dict:
                # Graph initializer
                self.initializer[param.variable_name] = param

                # Graph Inputs
                self.inputs[param.variable_name] =  param 
            else:
                # This condition will print whenever a parameter is not
                # found.
                # print("Not in: {}".format(param.variable_name))
                pass
        for iv in exe.data_variable:
            # Graph Inputs
            self.inputs[iv.variable_name] = iv
        for ov in exe.output_variable:
            # Only the final output of the graph is added
            self.outputs[ov.variable_name] = ov
        for gv in exe.generator_variable:
            # Graph Initializer
            self.initializer[gv.variable_name] = gv
            # Graph Inputs
            self.inputs[gv.variable_name] = gv
    
    def _set_nodes(self, func):
        """ Convert a function to a node or a group of nodes"""
        for f in self._net.function:
            node_name = f.name
            self.nodes[node_name] = f

    def create_graph(self):
        """ Retrieve network and extract information """
        net = self._set_network()
        self._set_shape_all()
        for f in net.function:
            self._set_nodes(f)

        # Broadcast target buffer
        self._set_variables()

class Exporter(ExprFunctor):
    """ A helper class for handling Relay expression copying from the Nnabla protobuf object.
    Parameters
    ----------
    nnp : nnabla.utils.converter.nnabla.importer.NnpImporter 
        The NNabla model to be converted from .nnp file, must contain
        the protobuf object.

    shape : dict of str to tuple, optional
        The input shape of the graph.

    dtype : str or dict of str to str
        The input types of the graph.
    
    batch_size : int 
        The batch size from the graph."""
    def __init__(self, nnp, shape, dtype, batch_size=1):
        # For Graph creation
        self._nnp = nnp 
        self._batch_size = batch_size
        self._graph = NNablaGraph(nnp, shape, dtype, batch_size)

        # For Relay convertion
        self._nodes = {}
        self._params = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape if shape else {}
        self._dtype = dtype

        # For Value infering
        self._temp_params = {}
        self._mod = None
        self._infer_simulated = True

        # For Quantization Pass
        self._quantized = False
        self._prev_node = None
        self._scale = {}

        super(Exporter, self).__init__()

    def _parse_array(self, param):
        """ Grab Nnabla parameter and return TVM NDArray """
        # TODO: Complete with dtype, for a start expect every type to be float32
        if isinstance(param, nnabla_pb2.Parameter):
            shape = infer_nnabla_shape(param.shape.dim)
            np_array = np.array(param.data, dtype="float32").reshape(tuple(shape))
        elif isinstance(param, (np.ndarray, np.generic)):
            # TODO: check datatype
            np_array =  param
        return _nd.array(np_array)
    
    def _parse_dtype(self, func):
        # TODO: Create dtype parser to pass the correct datatype to Relay
        return None
    
    def _update_quantized_params(self, node, quantizer, inputs, shapes):
        """ Helper function to replace full-precision weights with quantized
            weights obtained from QAT simulated values."""

        # Pop weight or bias in floating point
        self._params.pop(node.input[0])
        # Pop simulated weight or bias in floating point
        self._params.pop(node.output[0])
        assert (node.output[0] not in self._params \
                and node.input[0] not in self._params)

        # Update params dict with quantized value
        # Replace e.g.: conv1/quantized_conv/W by conv1/quantized_conv/W_q
        # in parameters
        self._params[node.output[0]] = quantizer._quantized_values

        # Pop Quantization operator from nodes
        # First pop the corresponding floating point node
        self._nodes.pop(quantizer._inputs)
        # Second pop the quantized node saved as floating point
        self._nodes.pop(quantizer._outputs)
        # Create new Relay Var with quantized parameter and save it
        # as node
        q_name = quantizer._outputs
        self._nodes[q_name] = new_var(q_name,
                                        shape=self._params[q_name].shape,
                                        dtype=self._params[q_name].dtype)
        # Final check
        assert node.output[0] in self._nodes

        # Inputs and Shapes should be modified too as they have changed
        inputs.replace(node.input[0], node.output[0], self._nodes[q_name])
        shapes.replace(node.input[0], node.output[0], shapes[node.input[0]])
    
    def _convert_operator(self, input_data, func, shapes):
        """ Convert NNabla operator into Relay Operator
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        input_data : dict of str
            Name of the inputs with its dimension shape

        func : nnabla_pb2.Function
            Function that describes a node. Contains the followinf information:
                -   name : Operator name
                -   type : Operator type
                -   inputs : input functions
                -   outputs : output functions
                -   param : Special attribute from each operator
        
        shapes : nnabla_pb2.Parameter
            Weights 

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        op_type = func.type
        if op_type in _convert_map:
            if self._quantized :
                _convert_map.update(qnn_nnabla._convert_map)
                sym = _convert_map[op_type](input_data, func, shapes)
            else:
                sym = _convert_map[op_type](input_data, func, shapes)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported for frontend NNabla.'.format(op_type))
        return sym

    def from_nnabla(self):
        """Construct Relay expression from NNabla graph.
        
        Nnabla graph is a protobuf object.
        
        Returns
        -------
        mod: tvm.IRModule
            The returned relay module
            
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # Create NNabla graph
        self._graph.create_graph()
        graph = self._graph
    
        # Convert dict of nnabla_pb2.shape into dict ot list/tuple
        self._shape = graph._var_dict

        # 1- parse network inputs or parameters to relay
        # 1.1 - Get parameters from graph initializer
        for init_param in graph.initializer:
            tmp_param = graph.initializer[init_param]

            assert init_param == tmp_param.variable_name
            self._params[tmp_param.variable_name] = self._parse_array(tmp_param)
            self._nodes[tmp_param.variable_name] = new_var(tmp_param.variable_name,
                                                            shape=self._params[init_param].shape,
                                                            dtype=self._params[init_param].dtype)

        # 1.2 - Get parameters from graph input
        for i in graph.inputs:
            i_name = graph.inputs[i].variable_name
            d_type = "float32" # Force datatype for now
            if i_name in self._params:
                # i is a param instead of an input
                self._num_param += 1
                self._params[i_name] = self._params.pop(i_name)
                self._nodes[i_name] = new_var(i_name,
                                              shape=self._params[i_name].shape,
                                              dtype=self._params[i_name].dtype)
            else:
                self._num_input += 1
                if i_name in self._shape:
                    tshape = list(self._shape[i_name])
                else:
                    raise ValueError("Must provide an input shape for `{0}`.".format(i_name))
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype= d_type
                assert isinstance(tshape, (list, tuple))
                self._nodes[i_name] = new_var(i_name, shape=tshape, dtype=dtype)

        # 2- get list of unsuppported ops
        unsupported_ops = set()
        for node in graph.nodes:
            op_name = graph.nodes[node].type 
            if op_name not in _convert_map and op_name != 'Constant':
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = 'The following operators are not supported for frontend NNabla: '
            msg += ', '.join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

        # 3- construct nodes, nodes are stored as directed acyclic graph
        for n in graph.nodes:
            op_type = graph.nodes[n].type
            op_name = graph.nodes[n].name

            node = graph.nodes[op_name]
            # Assert self._params type to be dictionary of str to tvm.nd.NDArray
            
            inputs = nnabla_input()
            shapes = nnabla_input()
            for i in node.input:
                inputs[i] = self._nodes[i]
                shapes[i] = self._shape[i]

            # Hardcoded Affine condition
            if op_type == 'Affine':
                # Nnabla affine operator requires output shape for
                # correct conversion
                shapes[node.output[0]] = self._shape[node.output[0]]
            if op_type == 'Constant':
                # Constant value initialized to 0.0
                # TODO: Check how to obtain Constant value from Nnabla
                # dtype = "int32" if self._quantized else "float32"
                dtype = "float32"
                constant_tensor = np.zeros(tuple(node.constant_param.shape.dim),
                                           dtype=dtype)
                self._num_param += 1 
                array = self._parse_array(constant_tensor)
                self._params[node.output[0]] = array
                self._nodes[node.output[0]] = new_var(
                    node.output[0],
                    shape=list(array.shape),
                    dtype=array.dtype)
            elif op_type in ['Pow2Quantize', 'FixedPointQuantize']:
                # TODO: In order to pass the simulated values to Relay in floating-point,
                # without transforming the values into the real quantized datatype,
                # we should comment the following flag. 
                # self._quantized = True

                # The quantizer for Conv2d and Affine operators quantizes
                # the weights and biases. 
                # In the case of activation quantization, quantization is applied 
                # after each Batch Normalization operator. This step should check 
                # which kind if quantization has to be performed.
                if (self._prev_node != None and self._prev_node.type \
                    in ['BatchNormalization', 'Add2']):
                    # Activation quantization after Batch Normalization
                    op = self._convert_operator(inputs, node, shapes)
                    node_output = node.output[0]
                    if not isinstance(op, _expr.TupleWrapper):
                        outputs_num = 1
                    else:
                        outputs_num = len(op)
                    assert len(node.output) == outputs_num, (
                        "Number of output mismatch {} vs {} in {}.".format(
                            len(node.output), outputs_num, op_name))
                    if outputs_num == 1:
                        self._nodes[node_output] = op
                    else:
                        for k, i in zip(list(node.output), range(len(node.output))):
                            self._nodes[k] = op[i]
                else:
                    # Weight and bias quantization for Convolution and Dense operator
                    quantizer = qnn_nnabla.QNNNode(node, self._params[node.input[0]], 
                                                self._params[node.output[0]])
                    
                    # Update parameters and nodes with quantized weights or bias
                    self._update_quantized_params(node, quantizer, inputs, shapes)

                # Save scale values to dequantize network after quantized operator
                # execution 
                if op_type == "FixedPointQuantize":
                    self._scale[node.output[0]] = node.fixed_point_quantize_param.delta
                else:
                # TODO: Complete with Pow2Quantize method
                    pass
            else:
                # TODO: Avoid passing scale or other quantization parameters as shapes 
                # because this will make the Relay Quantization Pass fail. This 
                # part of the code should be uncommented only for testing purposes when
                # weights or activations are converted into fixed-precision datatype.
                # for i in node.input:
                #     if i in self._scale:
                #         scale_name = i + "/delta"
                #         shapes[scale_name] = self._scale[i] 

                op = self._convert_operator(inputs, node, shapes)
                node_output = node.output[0]
                if not isinstance(op, _expr.TupleWrapper):
                    outputs_num = 1
                else:
                    outputs_num = len(op)
                assert len(node.output) == outputs_num, (
                    "Number of output mismatch {} vs {} in {}.".format(
                        len(node.output), outputs_num, op_name))
                if outputs_num == 1:
                    self._nodes[node_output] = op
                else:
                    for k, i in zip(list(node.output), range(len(node.output))):
                        self._nodes[k] = op[i]

            # Save node into pre_node, this node will only be used when quantization 
            # is applied. In order to check whether activation quantization or
            # weight/bias quantization should take place
            self._prev_node = node 

        # 4- return the outputs
        outputs = [self._nodes[i] for i in graph.outputs]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        func = _function.Function(analysis.free_vars(outputs), outputs)

        # Create pass for dequantization
        if self._quantized:
            pass

        return IRModule.from_expr(func), self._params 

def from_nnabla(model, shape=None, dtype="float32"):
    """Convert NNabla model to relay Function.

    Parameters
    ----------
    model : nnabla.utils.converter.nnabla.importer.NnpImporter 
        The NNabla model to be converted from .nnp file, must contain
        the protobuf object

    shape: dict of str to int list/tuple
        Input shapes of the model, optional

    dtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation.

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by Relay.
    """
    try:
        import nnabla as nn
        from nnabla.utils.converter.nnabla import NnpImporter
        
        # TODO: Check model
    except ImportError:
        raise ImportError("Nnabla must be installed!")
    
    if model is not None:
        network_name = model.protobuf.executor[0].network_name
    else:
        print("Import from {} failed.".format(model))
    
    mod, params = Exporter(model, shape, dtype).from_nnabla()
    nnabla_model = None 
    
    return mod, params 

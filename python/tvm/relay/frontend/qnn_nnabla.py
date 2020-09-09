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
""" Functions to convert simulated NNabla quantized models to QNN """
import sys
import logging
import numpy as np
logging.basicConfig(level=logging.CRITICAL)
import nnabla as nn
from nnabla.utils import nnabla_pb2

import tvm 
from tvm import relay 
from tvm.relay import expr as _expr 
from tvm.relay import op as _op
from tvm.relay.frontend.common import infer_shape, infer_channels
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform as _transform 
from tvm.ir import IRModule 
from .common import infer_type
from ... import nd as _nd
import pdb

# #############################################################################
# Helper functions
# ----------------

class QNNNode:
    """ A helper Class for weight quantization parameters.
    Something

    Parameters
    ----------
    func : nnabla_pb2.Function
        Graph node with quantization parameters
    
    values : NDarray 
        Full-precision original array (weight/bias)

    simulated_value : NDarray
        Quantized simulated value in full-precision. To convert to fixed-point value
        quantization methods should be executed.
    """

    def __init__(self, func, values=None, simulated_values=None):
        self._name = func.name
        self._qtype = func.type
        self._values = values
        self._quantized_values = simulated_values
        self._inputs = func.input[0]
        self._outputs = func.output[0]
        self._params = {}

        self._set_parameters(func)
        self._set_weight_quant_params(simulated_values)

    def _set_parameters(self, func):
        if self._qtype == 'Pow2Quantize':
            _bw = func.pow2_quantize_param.n
            self._params['sign'] = func.pow2_quantize_param.sign
            self._params['with_zero'] = func.pow2_quantize_param.with_zero
            self._params['ste'] = func.pow2_quantize_param.ste_fine_grained
            self._params['bitwidth'] = _bw
            
            _bound = func.pow2_quantize_param.m
            self._params['m'] = _bound if _bound != 0 else 1

            delta = find_delta(self._values, _bw)
            self._params['delta'] = delta
            xmax = delta * (2 ** (_bw - 1) - 1)
            xmax = 2 ** np.round(np.log2(xmax))
            xmin = xmax / 2 ** (2 ** (_bw - 1) - 1)

            w_xmax_max = 127
            w_xmax_min = 2**-8
            w_xmin_max = 127
            w_xmin_min = 2**-16

            xmin = np.clip(xmin, w_xmin_min + 1e-5, w_xmin_max - 1e-5)
            xmax = np.clip(xmax, w_xmax_min + 1e-5, w_xmax_max - 1e-5)

            self._params['xmax'] = xmax
            self._params['xmin'] = xmin

        elif self._qtype == 'FixedPointQuantize':
            _bw = func.fixed_point_quantize_param.n
            self._params['sign'] = func.fixed_point_quantize_param.sign
            self._params['ste'] = func.fixed_point_quantize_param.ste_fine_grained
            self._params['bitwidth'] = _bw
            self._params['delta'] = func.fixed_point_quantize_param.delta
            # TODO: Training stepsize can differ from inference calculated delta,
            #       avoid asserting stepsize for now.
            # assert self._params['delta'] != find_delta(self._params, _bw)

    def _set_weight_quant_params(self, values):
        """ Convert simulated params values into quantized params """
        tmp_values = values.asnumpy()
        if self._qtype == 'Pow2Quantize':
            # dtype = self._infer_datatype()
            # tmp_values = (-1 * np.sign(tmp_values) * np.log2(np.abs(tmp_values))).astype(dtype)
            self._quantized_values = _nd.array(tmp_values)

        elif self._qtype == 'FixedPointQuantize':
            # delta = self._params['delta']
            # dtype = self._infer_datatype()
            # tmp_values = (tmp_values * (delta ** (-1))).astype(dtype) 
            self._quantized_values = _nd.array(tmp_values)
        else:
            msg = '{} not implemented in TVM.'
            raise tvm.error.OpNotImplemented(msg.format(self._qtype))

    def _get_weight_quant_params(self):
        return self._quantized_values

    def _infer_datatype(self):
        bitwidth = self._params['bitwidth']
        sign = self._params['sign']
        if sign:
            if bitwidth == 8:
                dtype = 'int' + str(bitwidth)
            elif bitwidth == 32:
                dtype = 'int' + str(bitwidth)
            elif bitwidth == 64:
                dtype = 'int' + str(bitwidth)
            else:
                dtype = 'int8'
        else:
            if bitwidth == 8:
                dtype = 'uint' + str(bitwidth)
            elif bitwidth == 32:
                dtype = 'uint' + str(bitwidth)
            elif bitwidth == 64:
                dtype = 'uint' + str(bitwidth)
            else:
                dtype = 'uint8'
        return dtype

def find_delta(w, bw):
    """ Finds optimal quantization step size for FP quantization
    Parameters
    ----------
    w : NDarray or nnabla_pb2.Parameter
        params (could be weights, bias or output from other functions)

    bw : int or str
        Bitwidth for the quantization method

    Returns
    -------
    d : int or float
        Stepsize value
    """
    maxabs_w = np.max(np.abs(w.asnumpy())) + np.finfo(np.float32).eps 

    if bw > 4:
        return 2**(np.ceil(np.log2(maxabs_w/(2**(bw-1)-1))))
    else:
        return 2**(np.floor(np.log2(maxabs_w/(2**(bw-1)-1))))

def _convert_fixed_point_quantize():
    """ Function that retrieves the uniformly quantized values simulated during
        training. Should output the input value in fixed-point representation."""

    def _impl(inputs, func, shapes):
        # Function Parameters extracted from NNabla. Stepsize is 
        # pre-calculated during training and should be available as
        # a parameter
        _bw = func.fixed_point_quantize_param.n
        _ste = func.fixed_point_quantize_param.ste_fine_grained
        _delta = func.fixed_point_quantize_param.delta
        # For quantized ReLU, sign is ommited and variable is
        # not stored. Otherwise it will be stored with
        # the true value
        _sign = func.fixed_point_quantize_param.sign
        # import pdb 
        # pdb.set_trace()
        if _sign:
            _max = ((1 << (_bw - 1)) - 1.0) * _delta
            _min = -(_max)
        else:
            _max = ((1 << _bw) - 1.0) * _delta
            _min = 0.
        _input_shape = tuple(shapes[0])

        # inputs[0] = _op.cast(inputs[0], "float32")

        # Get original sign
        sign_val = _op.sign(inputs[0])
        # tmp_qi =  floor((|xi|*d**-1) + (2**-1))
        _tmp = _op.abs(inputs[0])
        _tmp = _op.divide(_tmp, _expr.const(_delta))
        _tmp = _op.add(_tmp, _expr.const(0.5))
        _tmp = _op.floor(_tmp)
        # qi = sign(xi) * tmp_qi
        out = _op.multiply(sign_val, _tmp)
        # For simulated quantized value, multiply but the stepsize,
        # otherwise comment following line
        out = _op.multiply(out, _expr.const(_delta))
        # Modify quantized values not to exceed the maximum and minimum
        # values
        _max_const = _op.full(_expr.const(_max), _input_shape)
        _min_const = _op.full(_expr.const(_min), _input_shape)
        out = _op.where((inputs[0] > _max_const), _max_const, out)
        out = _op.where((inputs[0] < _min_const), _min_const, out)
        # TODO: Test real quantization value
        # out = _op.cast(out, "int32")
        
        return out

    return _impl

def _convert_pow2_quantize():
    """ Function that retrieves the uniformly quantized values simulated during
        training. Should output the input value in fixed-point representation."""

    def _impl(inputs, func, shapes):
        # Function Parameters extracted from NNabla. Stepsize is 
        # pre-calculated during training and should be available as
        # a parameter
        _n = func.pow2_quantize_param.n
        _sign = func.pow2_quantize_param.sign
        _with_zero = func.pow2_quantize_param.with_zero
        _m = func.pow2_quantize_param.m

        _bw = _n - 1 if _sign else _n
        _bw = _bw -1 if _with_zero else _bw
        _input_shape = tuple(shapes[0])

        _ref_p_max = 2 ** _m
        _ref_p_min = 2 ** (_m - ((1 << _bw) - 1))
        _ref_pruning_threshold = _ref_p_min * (2. ** -0.5)


        _tmp = _op.round(_op.log2(_op.abs(inputs[0])))
        _tmp = _op.power(_expr.const(2.), _tmp)
        _ref_max_const = _op.full(_expr.const(_ref_p_max), _input_shape)
        _ref_min_const = _op.full(_expr.const(_ref_p_min), _input_shape)
        _tmp = _op.where((_tmp > _ref_max_const), _ref_max_const, _tmp)
        if _with_zero:
            _tmp_q = _op.copy(_tmp)
            _tmp_q = _op.where((_tmp_q < _ref_min_const), _ref_min_const, _tmp_q)
            _ref_pruning = _op.full(_expr.const(_ref_pruning_threshold), _input_shape)
            _zeros = _op.zeros(_input_shape, "float32")
            _tmp_q = _op.where((_op.abs(inputs[0]) < _ref_pruning), _zeros, _tmp)
        if not _with_zero:
            _tmp_q = _op.where((_tmp < _ref_min_const), _ref_min_const, _tmp)
        if _sign:
            out = _op.multiply(_op.sign(inputs[0]), _tmp)
        else:
            _zeros = _op.zeros(_input_shape, "float32")
            if _with_zero:
                out = _op.where((_op.sign(inputs[0]) < _zeros), _zeros, _tmp_q)
            else:
                out = _op.where((_op.sign(inputs[0]) < _zeros), _ref_min_const, _tmp_q)

        return out

    return _impl

def _convert_quantized_conv():
    def _impl(inputs, func, shapes):
        
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

        # Quantize inputs
        if func.input[0] == 'x':
            inputs[0] = _op.round(inputs[0])
            inputs[0] = _op.clip(inputs[0], a_min=-127.0, a_max=127.0)
            inputs[0] = _op.cast(inputs[0], 'int8')
        # else:
        #     # inputs[0] = _op.clip(inputs[0], a_min=-127.0, a_max=127.0)
        #     inputs[0] = _op.cast(inputs[0], 'int8')
        
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
                                 out_dtype="int32")
        
        use_bias = len(inputs) == 3
        if use_bias:
            inputs[2] = _op.cast(inputs[2], "int32")
            conv_out = _op.nn.bias_add(conv_out, inputs[2])

        return conv_out 

    return _impl

def _convert_quantized_gemm():
    def _impl(inputs, func, shapes):
        # Equivalent Op to GEMM in ONNX
        # Y = alpha * A * B + beta * C(If exists)
        
        # TODO: Infer values from NNabla Parameters
        alpha = int(1.0)
        beta = int(1.0)
        transA = 0
        transB = 0

        # get number of channels 
        channels = infer_channels(inputs[1], not transB)
        if transA:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not transB: 
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        inputs[0] = _op.nn.batch_flatten(inputs[0])
        
        if alpha != 1:
            inputs[0] *= _expr.const(alpha)
        out = _op.nn.dense(inputs[0], inputs[1], units=channels,
                           out_dtype='int32')
        
        use_bias = len(inputs) == 3
        if use_bias or (beta != 0):
            inputs[2] = _op.cast(inputs[2], "int32") 
            return _op.nn.bias_add(out, _expr.const(beta) * inputs[2])
        else:
            return out 

    return _impl

def _convert_quantized_affine():
    def _impl(inputs, func, shapes):
        base_axis = func.affine_param.base_axis
        x_shape = shapes[0]
        w_shape = shapes[1]
        y_shape = list(shapes[func.output[0]])
        x_shape_dims = [np.prod(x_shape[:base_axis]),
                        np.prod(x_shape[base_axis:])]
        gemm_output_shape = [np.prod(x_shape[:base_axis]),
                             np.prod(w_shape[1:])]

        # Reshape inputs
        if x_shape_dims != x_shape:
            inputs[0] = _op.reshape(inputs[0], x_shape_dims)

        # Quantize inputs
        def _quant(input):
            # input = _op.round(input)
            input = _op.clip(input, a_min=-7.0, a_max=7.0)
            input = _op.cast(input, 'int8')
            return input
        inputs[0] = _quant(inputs[0])
        inputs[1] = _quant(inputs[1])

        # GEMM
        if gemm_output_shape == y_shape:
            out = _convert_quantized_gemm()(inputs, func, shapes[:-1])
        else:
            raise tvm.error.OpAttributeInvalid(
                'Operator {} does not currently support different \
                output shapes'.format(func.type))

        out = _op.cast(out, "float32")

        return  out
    
    return _impl

def _convert_quantized_batchnorm():
    def _impl(inputs, func, shapes):

        # Dequantize output from convolution: int32 -> float32
        inputs[0] = _op.cast(inputs[0], "float32") 

        def _quantize(input, min_val=-127.0, max_val=127.0, dtype="int32"):
            input = _op.round(input)
            input = _op.clip(input, a_min=-127.0, a_max=127.0)
            input = _op.cast(input, "int32")
            
            return input

        attrs = {'beta': inputs[1],
                  'gamma': inputs[2],
                  'moving_mean': inputs[3],
                  'moving_var': inputs[4],
                  'axis': func.batch_normalization_param.axes[0],
                  'epsilon': np.ceil(func.batch_normalization_param.eps),
                  'center': True,
                  'scale': True}
  
        result, moving_mean, moving_var = _op.nn.batch_norm(inputs[0], **attrs)

        # Quantize batch_normalization output: float32 -> int32
        # result = _quantize(result, sys.maxsize, -sys.maxsize)

        return result

    return _impl

def _convert_quantized_elemwise():
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
            inputs[0] = _op.round(inputs[0])
            inputs[0] = _op.clip(inputs[0], a_min=-127.0, a_max=127.0)
            inputs[0] = _op.cast(inputs[0], "int8")
            result = op_map[op_name](inputs[0], inputs[1])

        return result
    return _impl
# Converter map with custom operators
# Include quantization operators or special operators designed
# for tests in here.
# In case of quantization aware inference, this map will update the
# full-precision defined methods.
_convert_map = {
    # 'FixedPointQuantize'       : _convert_fixed_point_quantize(),
    # 'Pow2Quantize'             : _convert_pow2_quantize(),
    'Add2'                     : _convert_quantized_elemwise(),
    'Convolution'              : _convert_quantized_conv(),
    'Affine'                   : _convert_quantized_affine(),
    'BatchNormalization'       : _convert_quantized_batchnorm()
}        
###############################################################################
# AVOID THIS SECTION OF CODE, IT WAS DEFINED TO TEST MUTATORS. 
# FOR FPGA DEPLOYMENT, MUTATORS SHOULD BE CONSIDERED AS SEVERAL TRANSFORMATIONS
# HAVE TO BE IMPLEMENTED BEFORE AND DURING QUANTIZATION.
# FOR REFERENCE CHECK: 
# https://tvm.apache.org/docs/vta/tutorials/optimize/convolution_opt.html#sphx-glr-vta-tutorials-optimize-convolution-opt-py
###############################################################################
# def cast_fixedpoint(func):
#     """Cast from fp32 to [int8, int32]
    
#     Parameters 
#     ---------
#     func: Function
#         The original graph.
    
#     Returns
#     -------
#     The graph after casting to fixed-point integer.
#     """
#     # Convolution and affine operators are already quantized
#     filter_list = ['nn.conv2d']
#     class DowncastMutator(ExprMutator):
#         """ Cast to fixed-point mutator """
#         def visit_call(self, call):
#             dtype = 'float32' if call.op.name in filter_list else 'int32'
#             new_fn = self.visit(call.op)
#             # Collect the original dtypes
#             type_list = []
#             if call.op.name in filter_list:
#                 #   TODO: Complete with convolution and affine
#                 for arg in call.args:
#                     if isinstance(arg, TupleGetItem) and isinstance(arg.tuple_value, Call):
#                             tuple_types = arg.tuple_value.checked_type.fields
#                             type_list.append(tuple_types[arg.index].dtype)
#                 if call.op.name == 'vision.get_valid_counts':
#                     tuple_types = call.checked_type.fields
#                     for cur_type in tuple_types:
#                         type_list.append(cur_type.dtype)
            
#             args = [self.visit(arg) for arg in call.args]
#             new_args = list()
#             arg_idx = 0
#             for arg in args:
#                 if isinstance(arg, (Var, Constant)):
#                     new_args.append(cast(arg, dtype=dtype))
#                 else:
#                     if call.op.name in filter_list:
#                         if isinstance(arg, TupleGetItem) and type_list[arg_idx] == 'int32':
#                             new_args.append(arg)
#                         else:
#                             new_args.append(cast(arg, dtype=dtype))
#                     else:
#                         new_args.append(arg)
#                 arg_idx += 1
#             if call.op.name in filter_list and call.op.name != 'nn.conv2d':
#                 return cast(Call(new_fn, new_args, call.attrs), dtype='int32')
#             return Call(new_fn, new_args, call.attrs)
    
#     class UpcastMutator(ExprMutator):
#         """ Upcast output back to fp32 mutator """
#         def visit_call(self, call):
#             return cast(call, dtype='float32')

#     def infer_type(expr):
#         """ Method to infer the type of an intermediate node in the relay graph """
#         mod = IRModule.from_expr(expr)
#         mod = _transform.InferType()(mod)
#         entry = mod['main']
#         return entry if isinstance(expr, Function) else entry.body

#     func = infer_type(func)
#     downcast_pass = DowncastMutator()
#     func = downcast_pass.visit(func)
#     upcast_pass = UpcastMutator()
#     func = upcast_pass.visit(func)
#     func = infer_type(func)

#     return func 



    


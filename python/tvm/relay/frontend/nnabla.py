""" NNabla: Neural Network Libraries Frontend for Relay """
import numpy as np
import collections
import nnabla as nn
from nnabla.utils.nnp_graph import NnpLoader, FunctionProto, VariableProto
from nnabla.utils import nnabla_pb2
from nnabla.parameter import get_parameter_or_create, save_parameters, get_parameter_or_create
import google.protobuf.text_format as text_format 
import zipfile
import shutil
import tempfile
import os
import attr  
import sys 
import pdb

import tvm 
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis 
from .. import expr as _expr
from .. import function as _function
from .. import op as _op 

from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_shape, infer_channels
from .common import infer_type, get_name
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated 

__all__ = ['from_nnabla']

# #############################################################################
# Helper functions
# ----------------
# 

# TENSOR_TYPE_TO_DTYPE = {
#     TensorProto.FLOAT: np.float32,
#     TensorProto.BOOL: np.bool,
#     TensorProto.UINT8: np.uint8,
#     TensorProto.INT8: np.int8,
#     TensorProto.INT32: np.uint32,
#     TensorProto.INT32: np.int32,
#     TensorProto.INT64: np.int64,
# }

def load_nnp(nnp_file):
    """ Add description """
    def load_nnp(nnp_path):
    # Load nnp file
    
    return NnpImporter(nnp_path, expad_network=False, executor_index=True).execute()

def default_layout(dims):
    """
    A helper function to get default layout 
    """
    if dims == 1:
        return 'NCW'
    elif dims == 2:
        return 'NCHW'
    elif dims == 3:
        return 'NCDHW'

    msg = "Only 1d, 2d and 3d layouts are currently supported"
    raise tvm.error.OpAttributeInvalid(msg.format(op_name))

def  dimension_picker(prefix, suffix=''):
    """ Check that dimensions are supported """
    # TODO: Check variables names
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 1:
            return prefix + '1d' + suffix 
        if len(kernel) == 2:
            return prefix + '2d' + suffix 
        if len(kernel) == 3:
            return prefix + '3d' + suffix 
        msg = 'Only 1D, 2D and 3D kernels are supported for operator {}.'
        op_name = prefix + '1d/2d/3d'
        raise tvm.error.OpAttributeInvalid(msg.format(op_name))

def dimension_constraint():
    """ A helper function to restric dimensions """
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) in [1, 2, 3]:
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
    return out_shape

# def get_tensor_type(name, type_dict):
#     if name in type_dict:
#         return type_dict[name]
#     else:
#         # Default tensor type to float
#         return TensorProto.FLOAT


# #############################################################################
# Operator definition
# -------------------
# 
# Each NNabla operator has its own converter 

def _none():
    def _impl(inputs, func):
        return None 
    return _impl 

def _convert_reshape():
    def _impl(inputs, func):
        if hasattr(func, 'shape'):
            return _op.reshape(inputs[0], func.shape)
        else:
            raise NotImplementedError("Yet to support dynamic input case")
            # return _op.reshape(inputs[0], inputs[1])
    return _impl

def _convert_concat():
    def _impl(inputs, func):
        return _op.concatenate(inputs, axis=func.axis)
    return _impl

def _convert_relu():
    def _impl(inputs, func):
        data = inputs[0]
        return _op.nn.relu(data)
    return _impl 

def _convert_convolution():
    def _impl(inputs, func):
        # TODO: Map layouts
        # TODO: Check for all possible input combinations
        # for stride, pads, dilation, groups, channels, kernel_size 
        data_layout = "NCHW"
        kernel_layout = "OIHW"

        conv_out = _op.nn.conv2d(inputs[0],
                                 inputs[1],
                                 strides=(func.sy, func.sx),
                                 padding=(func.ph, func.pw, func.ph, func.pw),
                                 dilation=(func.dy, func.dx),
                                 groups=func.groups,
                                 channels=func.inputs[1].shape[0],
                                 kernel_size=func.inputs[1].shape[2:],
                                 data_layout=data_layout,
                                 kernel_layout=kernel_layout,
                                 out_layout="",
                                 out_dtype="")
        
        use_bias = len(inputs) == 3

        if use_bias:
            return _op.nn.bias_add(conv_out, inputs[2])
        else:
            return conv_out 
    
    return _impl

def _linear():
    def _impl(inputs, func):
        # Equivalent Op to GEMM in ONNX
        # Y = alpha * A * B + beta * C(If exists)
        alpha = float(1.0)
        beta = float(1.0)

        # get number of channels 
        channels = infer_channels(inputs[1])
        inputs[0] = _op.nn.batch_flatten(inputs[0])
        out = _op.nn.dense(_expr.const(alpha) * inputs[0],
                           inputs[1], units=channels)
        
        use_bias = len(inputs) == 3

        if use_bias:
            return _op.nn.bias_add(out, _expr.const(beta) * inputs[2])
        else:
            return out 

    return _impl

# def _convert_softmax():
 

# def _convert_pooling():

# def _convert_batchnorm():

# def _convert_elemwise():

# def _convert_flatten():


# #############################################################################
# Converter map for NNabla 
# ------------------------
# 
# NNabla operators linked to the Relay converter
# 

_convert_map = {
    'SoftMax'                  : _convert_softmax,
    'ReLU'                     : _convert_relu,
    'LeakyReLU'                : _none,
    'PReLU'                    : _none,
    'ELU'                      : _none,

    'AveragePooling'           : _convert_pooling,
    'MaxPooling'               : _convert_pooling,
    'GlobalAveragePooling2D'   : _none,
    'GlobalMaxPooling2D'       : _none,
    'Convolution'              : _convert_convolution,
    'Conv2DTranspose'          : _none,
    'DepthwiseConv2D'          : _none,

    'Flatten'                  : _convert_flatten,
    'Reshape'                  : _convert_reshape,
    'Concatenate'              : _convert_concat,
    'BatchNormalization'       : _convert_batchnorm,
    'Add2'                     : _convert_elemwise
}

# #############################################################################
# NNabla converter definiton
# --------------------------

def get_converter(op):
    """ Convert NNabla operators to Relay Converter """
    return _convert_map[op]

def _check_unsupported_layers():
    # TODO: Complete with missing layers

# def nnabla_op_to_relay(inexpr, nnabla_layer, outname, etab):
#     """Convert a NNabla layer to a Relay expression and update the expression table.

#     Parameters
#     ----------
#     inexpr : relay.expr.Expr or a list of it
#         The input Relay expression(s).

#     NNabla_layer : NNabla.layers
#         The NNabla layer to be converted.

#     outname : str
#         Name of the output Relay expression.

#     etab : relay.frontend.common.ExprTable
#         The global expression table to be updated.
#     """
#     op_name = # TODO: get function name
#     if op_name not in _convert_map:
#         raise tvm.error.OpNotImplemented(
#             'Operator {} is not supported for frontend Nnabla'.format(op_name))
#     outs = _convert_map[op_name](inexp, etab)
#     outs = _as_lists(outs)
#     for t_idx, out in enumarete(outs):
#         name = outname + ":" + str(t_idx)
#         etab.set_expr(name, out)

class NNablaGraph(object):
    def __init__(self, nnp, batch_size=1, shape, dtype):
        # NNabla related variables
        self._nnp = nnp.protobuf        # nnabla graph as protobuf object
        self._batch_size = batch_size   # executor batch_size
        self._net = None                # network_name
        self._executor = None
        self._parameters = {}
        self.var_dict = {}
        self.initializer = {}
        self.inputs = {}
        self.outputs = {}
        self.nodes_ = {}

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
        bs = self.batch_size
        if bs < 0:
            bs = self._net._batch_size
        self._batch_size = bs
        # store all variable shape info to use later 
        for v in self._net.variable:
            self._shape[v.name] = replace_negative_size_with_batch_size(
                v.shape, bs)
        
        for p in self._nnp.parameter:
            self._parameters[p.variable_name] = p
    
    def _set_variables(self):
        exe = self._executor
        for param in self._nnp.parameter:
            if param.variable_name in self.var_dict:
                # Graph initializer
                self.initializer[param.variable_name] = param 

                # Graph Inputs
                self.inputs[param.variable] =  param 
        
            else:
                print("Not in: {}".format(param.variable_name))

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
            node_name = f.type
            self.nodes[node_name] = f

    def create_graph(self):
        net = self.set_network()
        self.set_shape_all()
        for f in net.function:
            self.set_nodes(f)

        # Broadcast target buffer

        self.set_variables()

class Exporter(ExprFunctor):
    """ Add information """
    def __init__(self, nnp, batch_size=1, shape, dtype):
        # For creating Graph 
        self._nnp = nnp 
        self._batch_size = batch_size

        # For Relay convertion
        self._nodes = {}
        self._params = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape if shape else {}
        self._dtype = dtype

        # For infering Values
        self._temp_params = {}
        self._mod = None
        self._infer_simulated = True 
        super(Exporter, self).__init__()

    def _parse_array(self, param):
        """ Grab Nnabla parameter and return TVM NDArray """
        # TODO: Complete with dtype, for a start expect every type to be float32
        np_array = np.array(param.data, dtype="float32").reshape(tuple(param.shape.dim))
        
        return _nd.array(np_array)
    
    def _parse_dtype(self, func):
        """ TODO: Create dtype parser to pass the correct datatype to Relay """
            

    
    
    def _convert_operator(self, input_data, func):
        """ Convert NNabla operator into Relay Operator
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        input_data : nnabla_pb2.Parameter
            Weights

        func : nnabla_pb2.Function
            Function that describes a node. Contains the followinf information:
                -   name : Operator name
                -   type : Operator type
                -   inputs : input functions
                -   outputs : output functions
                -   param : Special attribute from each operator

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        op_name = func.type 
        if op_name in _convert_map:
            sym = _convert_map[op_name](input_data, func)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported for frontend NNabla.'.format(op_name))
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
        graph = Graph(self._nnp, self._batch_size, self._shape, self._dtype).create_graph()
        self._shape = graph.var_dict

        # 1- parse network inputs or parameters to relay
        # 1.1 - Get parameters from graph initializer
        for init_param in graph.initializer:
            self._params[init_param.variable_name] = self._parse_array(init_param)
            self._nodes[init_param.variable_name] = new_var(init_param.variable_name,
                                                            shape=self._params[init_param].shape,
                                                            dtype=self._params[init_param].dtype)

        # 1.2 - Get parameters from graph input
        for i in graph.inputs:
            i_name = i.variable_name
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
                    tshape = self._shape[i_name]
                else:
                    raise ValueError("Must provide an input shape for `{0}`.".format(i_name))
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype= d_type
                self._nodes[i_name] = new_var(i_name, shape=tshape, dtype=dtype)

        # 2- get list of unsuppported ops
        unsupported_ops = set()
        for node in graph.nodes:
            op_name = node.type 
            if op_name not in convert_map and op_name != 'Constant':
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = 'The following operators are not supported for frontend NNabla: '
            msg += ', '.join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

        # 3- construct nodes, nodes are stored as directed acyclic graph
        for node in graph.node:
            op_name = node.type

            # inputs =  # Define input list or dict 
            op = self._convert_operator(self._params, node)
            node_output = self._fix_outputs(op_name, node.output)
            if not isinstance(op, _expr.TupleWrapper):
                outputs_num = 1
            else:
                outputs_num = len(op)
            assert len(node_output) == outputs_num, (
                "Number of output mismatch {} vs {} in {}.".format(
                    len(node_output), outputs_num, op_name))
            if outputs_num == 1:
                self._nodes[node_output[0]] = op
            else:
                for k, i in zip(list(node_output), range(len(node_output))):
                    self._nodes[k] = op[i]
            

        # 4- return the outputs
        outputs = [self._nodes[i.variable_name] for i in graph.outputs]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        func = _function.Function(analysis.free_vars(outputs), outputs)
        
        return IRModule.from_expr(func), self._params 

def from_nnabla(model, shape=None, dtype="float32"):
    """Convert NNabla model to relay Function.

    Parameters
    ----------
    model : NNabla.Nnp 
        The NNabla model to be converted in .nnp file, must contain
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
    nnp = load_nnp(model)
    if nnp is not None:
        network_name = nnp.protobuf.executor[0].network_name
    else:
        print("Import from {} failed.".format(model))
    
    mod, params = Exporter(nnp, shape, dtype).from_nnabla
    nnabla_model = None 
    
    return mod, params 






    





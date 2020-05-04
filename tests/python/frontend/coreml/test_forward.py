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
import numpy as np

from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import datatypes

import tvm
from tvm import te
from tvm.contrib import graph_runtime
import topi
import topi.testing
from tvm import relay
from tvm.relay.testing.config import ctx_list
from topi.testing import conv2d_nchw_python

import coremltools as cm
import model_zoo

def get_tvm_output(func, x, params, target, ctx,
                   out_shape=(1, 1000), input_name='image', dtype='float32'):
    with relay.transform.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_name, tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    m.run()
    # get outputs
    out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
    return out.asnumpy()

def run_model_checkonly(model_file, model_name='', input_name='image'):
    model = cm.models.MLModel(model_file)
    x = model_zoo.get_cat_image()
    shape_dict = {input_name : x.shape}
    # Some Relay passes change operators on the fly. Ensuring that we generate
    # new graph for each target.
    for target, ctx in ctx_list():
        mod, params = relay.frontend.from_coreml(model, shape_dict)
        tvm_output = get_tvm_output(mod["main"], x, params, target, ctx)
        print(target, ctx, model_name, 'prediction id: ', np.argmax(tvm_output.flat))

def test_mobilenet_checkonly():
    model_file = model_zoo.get_mobilenet()
    run_model_checkonly(model_file, 'mobilenet')

def test_resnet50_checkonly():
    model_file = model_zoo.get_resnet50()
    run_model_checkonly(model_file, 'resnet50')

def run_tvm_graph(coreml_model, target, ctx, input_data, input_name, output_shape, output_dtype='float32'):
    """ Generic function to compile on relay and execute on tvm """
    if isinstance(input_data, list):
        shape_dict = {}
        dtype_dict = {}
        for i, e in enumerate(input_name):
            shape_dict[e] = input_data[i].shape
            dtype_dict[e] = input_data[i].dtype
    else:
        shape_dict = {input_name: input_data.shape}
        dtype_dict = {input_name: input_data.dtype}

    mod, params = relay.frontend.from_coreml(coreml_model, shape_dict)
    with relay.transform.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_name):
            m.set_input(e, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_name, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        if not output_shape:
            tvm_output = m.get_output(0)
        else:
            tvm_output = m.get_output(0, tvm.nd.empty((output_shape), output_dtype))
        return tvm_output.asnumpy()

def verify_AddLayerParams(input_dim, alpha=2):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.add(a_np1, a_np2) + alpha
    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Add',
                            alpha=alpha,
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='ADD')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np1, a_np2], ['input1', 'input2'], b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_AddLayerParams():
    verify_AddLayerParams((1, 2, 2), 0)
    verify_AddLayerParams((1, 2, 2), 1)
    verify_AddLayerParams((1, 3, 3), 2)

def verify_MultiplyLayerParams(input_dim, alpha):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.multiply(a_np1, a_np2) * alpha
    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Mul',
                            alpha=alpha,
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='MULTIPLY')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np1, a_np2], ['input1', 'input2'], b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_MultiplyLayerParams():
    verify_MultiplyLayerParams((1, 2, 2), 0)
    verify_MultiplyLayerParams((1, 2, 2), 1)
    verify_MultiplyLayerParams((1, 3, 3), 2)

def verify_ConcatLayerParams(input1_dim, input2_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input1_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input2_dim).astype(dtype)

    b_np = np.concatenate((a_np1, a_np2), axis=1)
    inputs = [('input1', datatypes.Array(*input1_dim)),
              ('input2', datatypes.Array(*input2_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Concate',
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='CONCAT')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np1, a_np2], ['input1', 'input2'], b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_ConcatLayerParams():
    verify_ConcatLayerParams((1, 1, 2, 2), (1, 2, 2, 2))
    verify_ConcatLayerParams((1, 2, 4, 4), (1, 3, 4, 4))

def verify_UpsampleLayerParams(input_dim, scale, mode):
    dtype = "float32"

    a_np = np.full(input_dim, 1, dtype=dtype)
    if mode == 'NN':
        b_np = topi.testing.upsampling_python(a_np, (scale, scale))
    else:
        new_h = input_dim[2] * scale
        new_w = input_dim[3] * scale
        b_np = topi.testing.bilinear_resize_python(a_np, (new_h, new_w), 'NCHW')

    input = [('input', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input, output)
    builder.add_upsample(name='Upsample',
                         scaling_factor_h=scale,
                         scaling_factor_w=scale,
                         mode=mode,
                         input_name='input',
                         output_name='output')

    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, a_np, 'input', b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_UpsampleLayerParams():
    verify_UpsampleLayerParams((1, 16, 32, 32), 2, 'NN')
    verify_UpsampleLayerParams((1, 4, 6, 6), 3, 'BILINEAR')

def verify_l2_normalize(input_dim, eps):
    dtype = "float32"

    a_np = np.random.uniform(size=input_dim).astype(dtype)
    b_np = topi.testing.l2_normalize_python(a_np, eps, 1)

    input = [('input', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input, output)
    builder.add_l2_normalize(name='L2', epsilon=eps, input_name='input', output_name='output')

    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, a_np, 'input', b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_l2_normalize():
    verify_l2_normalize((1, 3, 20, 20), 0.001)

def verify_lrn(input_dim, size, bias, alpha, beta):
    dtype = "float32"
    axis=1
    a_np = np.random.uniform(size=input_dim).astype(dtype)
    b_np = topi.testing.lrn_python(a_np, size, axis, bias, alpha, beta)

    input = [('input', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(input, output)
    builder.add_lrn(name='LRN',
                    input_name='input',
                    output_name='output',
                    alpha=alpha,
                    beta=beta,
                    k=bias,
                    local_size=size)

    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, a_np, 'input', b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_lrn():
    verify_lrn((1, 3, 10, 20), 3, 1.0, 1.0, 0.5)

def verify_average(input_dim1, input_dim2, axis=0):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim1).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim2).astype(dtype)

    b_np = np.mean((a_np1, a_np2), axis=axis)

    inputs = [('input1', datatypes.Array(*input_dim1)),
              ('input2', datatypes.Array(*input_dim2))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='MEAN',
                            input_names=['input1', 'input2'],
                            output_name='output',
                            mode='AVE')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np1, a_np2], ['input1', 'input2'], b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_average():
    verify_average((1, 3, 20, 20), (1, 3, 20, 20))
    verify_average((3, 20, 20), (1, 3, 20, 20))
    verify_average((20, 20), (1, 3, 20, 20))

def verify_max(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.max((a_np1, a_np2, a_np3), axis=0)

    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim)),
              ('input3', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Max',
                            input_names=['input1', 'input2', 'input3'],
                            output_name='output',
                            mode='MAX')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np1, a_np2, a_np3],
                            ['input1', 'input2', 'input3'], b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_max():
    verify_max((1, 3, 20, 20))
    verify_max((20, 20))

def verify_min(input_dim):
    dtype = 'float32'

    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    b_np = np.min((a_np1, a_np2, a_np3), axis=0)

    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim)),
              ('input3', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_elementwise(name='Min',
                            input_names=['input1', 'input2', 'input3'],
                            output_name='output',
                            mode='MIN')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np1, a_np2, a_np3],
                            ['input1', 'input2', 'input3'], b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_min():
    verify_min((1, 3, 20, 20))
    verify_min((20, 20))

def verify_image_scaler(input_dim, blue_bias=0.0, green_bias=0.0, red_bias=0.0, image_scale=1.0):
    dtype = 'float32'
    a_np = np.random.uniform(size=input_dim).astype(dtype)
    # make sure it is valid image format CHW.
    assert len(a_np.shape) == 3 and a_np.shape[0] == 3
    b_np = np.zeros(a_np.shape, dtype=dtype)
    b_np[0, :, :] = image_scale * a_np[0, :, :] + blue_bias
    b_np[1, :, :] = image_scale * a_np[1, :, :] + green_bias
    b_np[2, :, :] = image_scale * a_np[2, :, :] + red_bias
    b_np = np.add(a_np, b_np)
    inputs = [('input1', datatypes.Array(*input_dim)),
              ('input2', datatypes.Array(*input_dim))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.set_pre_processing_parameters(image_input_names=['input1'],
                                          is_bgr=True,
                                          blue_bias=blue_bias,
                                          green_bias=green_bias,
                                          red_bias=red_bias,
                                          image_scale=image_scale)
    # add one add layer to make CoreML model format valid
    # add layer has been tested before.
    builder.add_elementwise(name='add', input_names=['input1', 'input2'],
                            output_name='output', alpha=0, mode='ADD')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np, a_np],
                            ['input1', 'input2'], b_np.shape, dtype)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_image_scaler():
    verify_image_scaler((3, 224, 224), image_scale=0.17)
    verify_image_scaler((3, 224, 224),
                        blue_bias=-1.7669800519943237,
                        green_bias=-1.985260009765625,
                        red_bias=-2.102560043334961,
                        image_scale=0.379)

def verify_convolution(input_dim, filter, padding):
    dtype = 'float32'
    N, C, H, W = input_dim
    OC, _, KH, KW = filter
    a_np = np.random.uniform(size=input_dim).astype(dtype)
    w_np = np.random.uniform(size=(OC, C, KH, KW)).astype(dtype)
    w_np_cm = np.transpose(w_np, axes=(2, 3, 1, 0))
    b_np = conv2d_nchw_python(a_np, w_np, [1, 1], padding)
    inputs = [('input1', datatypes.Array(C, H, W))]
    output = [('output', datatypes.Array(*b_np.shape))]
    builder = NeuralNetworkBuilder(inputs, output)
    builder.add_convolution(name='conv', kernel_channels=3, output_channels=OC,
                            height=KH, width=KW, stride_height=1, stride_width=1,
                            border_mode=padding.lower(), groups=1,
                            W=w_np_cm, b=None, has_bias=False,
                            is_deconv=False,
                            input_name='input1',
                            output_name='output')
    model = cm.models.MLModel(builder.spec)
    for target, ctx in ctx_list():
        out = run_tvm_graph(model, target, ctx, [a_np],
                            ['input1'], output_shape=None)
        tvm.testing.assert_allclose(out, b_np, rtol=1e-5)

def test_forward_convolution():
    verify_convolution((1, 3, 224, 224), filter=(32, 3, 3, 3), padding='VALID')
    verify_convolution((1, 3, 224, 224), filter=(32, 3, 3, 3), padding='SAME')

if __name__ == '__main__':
    test_forward_AddLayerParams()
    test_forward_ConcatLayerParams()
    test_forward_MultiplyLayerParams()
    test_forward_UpsampleLayerParams()
    test_forward_l2_normalize()
    test_forward_lrn()
    test_forward_average()
    test_forward_max()
    test_forward_min()
    test_mobilenet_checkonly()
    test_resnet50_checkonly()
    test_forward_image_scaler()
    test_forward_convolution()

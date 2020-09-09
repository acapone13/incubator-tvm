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
import tvm 
from tvm import te 
from tvm import relay 
from tvm.contrib import graph_runtime 
from tvm.relay.testing.config import ctx_list
from tvm.relay.frontend.nnabla import load_nnp as load_model 

import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.ext_utils import get_extension_context 
from nnabla.utils.nnp_graph import NnpLoader 

def verify_nnabla_frontend(model, input_data, ctxl=ctx_list()):
    """ Assert that the output of a compiled model matches with that of its
    baseline."""

    def get_nnabla_outputs(x, dtype='float32'):




    def get_tvm_outputs(x, target, ctx, dtype='float32'):
        shape_dict = {name: x.shape for (name, x) in nnabla.inputs}
        mod, params = relay.frontend.from_nnabla(model, shape_dict)
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(mod,
                                             target,
                                             params=params)
        m = graph_runtime.create(graph, lib, ctx)
        for name, x 

        m.set_input(**params)
        m.run()
        return [m.get_output(i).asnumpy() for i in range(m.get_num_outputs())]
    
    x = input_data 
    nnabla_out = get_nnabla_outputs(x)
    nnabla_out = nnabla_out if isinstance(nnabla_out, list) else [kera_out]
    for target, ctx in ctx_lists():
        tvm_out = get_tvm_output(inputs, target, ctx)
        for nout, tout in zip(nnabla_out, tvm_out):
            tvm.testing.assert_allclose(nout, tout, rtol=1e-5, atol=1e-5)

# Full model test
def test_resnet():


# Single operator tests
def test_conv():

    def verify_conv():
        

    # Convolution with padding
    # Convolution without padding
    # Convolution with non uniform stride
    # Convolution with dilation

def test_activation():

    def verify_activation():
        # Add verification

def test_pooling():

    def verify_pooling():
        # Add verification

def test_elemwise():

    def verify_elemwise():
        # Add verification

def test_flatten():

    def verify_flatten():
        # Add verification

def test_reshape():

    def verify_reshape():
        # Add verification

def test_concat():

    def verify_concat():
        # Add verification

def test_batchnorm():

    def verify_batchnorm():
        # Add verification

def test_affine():

    def verify_affine():
        # Add verification

if __name__ == '__main__':
    
    # Single operator tests 
    test_conv()
    test_activation()
    test_pooling()
    test_elemwise()
    test_flatten()
    test_reshape()
    test_concat()
    test_batchnorm()
    test_affine()

    # Full model test
    test_resnet()


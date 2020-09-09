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
"""
Compile NNabla Models
===================
**Author**: `Augusto Capone`_

This article is an introductory tutorial to deploy NNabla models with Relay.

For us to begin with, NNabla package must be installed.

A quick solution is to install protobuf compiler, and

.. code-block:: bash

    pip install nnabla --user


or please refer to offical sites.
https://github.com/sony/nnabla 
https://nnabla.org/
"""
import numpy as np 
import logging  
import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(level=logging.CRITICAL)
import nnabla as nn
from nnabla.utils.nnp_graph import NnpLoader

import tvm
from tvm import relay 
from tvm.contrib import graph_runtime, util
from tvm.contrib.download import download_testdata
from tvm.relay.frontend.nnabla import load_nnp as load_model

###############################################################################
# Load pretrained NNabla model
# ----------------------------
# The example model used here is the cifar-10 image classification problem 
# with residual neural networks
# https://github.com/sony/nnabla-examples/tree/master/cifar10-100-collection
# we skip the training procedures, and download the saved .nnp model
model_url = 'https://github.com/acapone13/web_data/raw/master/resnet20_v1.nnp'
model_path = download_testdata(model_url, 'resnet20_v1.nnp', module='nnabla')
# now we have resnet20_v1.nnp on disk
nnabla_model = load_model(model_path)

###############################################################################
# Load a test image
# -----------------
# The famous example cat image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
image = Image.open(img_path).resize((32, 32))
plt.imshow(image)
image = np.array(image) - np.array([125.3, 123.0, 113.9])
image /= np.array([63.0, 62.1, 66.7])
image = image.transpose((2, 0, 1))
image = image[np.newaxis, :]
image = np.repeat(image, 1, axis=0)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

###############################################################################
# Compile the model with relay
# ----------------------------
# First we load the downloaded model and then we compile the graph
shape_dict = {'x': image.shape}
mod, params = relay.frontend.from_nnabla(nnabla_model, shape_dict)
## we want a probability so add a softmax operator
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

# Uncomment to see the IR of the Graph
# print(func)

target = 'llvm'

with tvm.transform.PassContext(opt_level=4):
    lib = relay.build(func, target, params=params)

##############################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we reproduce the same forward computation using TVM

ctx = tvm.cpu(0)
dtype = 'float32'
m = graph_runtime.GraphModule(lib['default'](ctx))
# set inputs
m.set_input('x', tvm.nd.array(image.astype(dtype)))
# execute graph
m.run()
# get outputs
tvm_output = m.get_output(0)
tvm_out_nnabla = tvm_output.asnumpy()[0]

# Timer to measure the execution time
timer = m.module.time_evaluator("run", ctx, number=4, repeat=3)

tcost = timer()
std = np.std(tcost.results) * 1000
mean = tcost.mean * 1000

# Print results and statistics
ind = np.argmax(tvm_output.asnumpy(), axis=1).astype('int')
print('The input [%s] picture was classified as [%s] with probability [%0.4f] in %0.2fs.'%
(class_names[3], class_names[ind[0]], tvm_output.asnumpy()[0][ind[0]], mean))






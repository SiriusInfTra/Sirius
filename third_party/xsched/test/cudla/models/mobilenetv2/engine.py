# from https://github.com/wang-xinyu/tensorrtx/blob/master/mobilenet/mobilenetv2/mobilenet_v2.py

import os
import sys
import struct
import argparse

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000
INPUT_BLOB_NAME = "input"
OUTPUT_BLOB_NAME = "output"
EPS = 1e-5

ENGINE_DIR = "./engines"
ENGINE_PATH = f"{ENGINE_DIR}/mobilenetv2.engine"
WEIGHT_PATH = "./mobilenetv2.wts"

os.makedirs(ENGINE_DIR, exist_ok=True)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
BUILD_LOG = open(f"{ENGINE_DIR}/build.log", "w")

class RndInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        trt.IInt8EntropyCalibrator2.__init__(self)
    
    # def get_batch(self, names):
    #     return None

    def get_batch_size(self):
        return 1
    
    # def read_calibration_cache(self):
    #     return None

    # def write_calibration_cache(self, cache):
    #     pass

slice_idx = 0
slice_layers = []
slice_tensors = {}
slice_config = None
slice_network = None
slice_builder = None

def add_slice():
    global slice_idx
    global slice_layers
    global slice_tensors
    global slice_config
    global slice_network
    global slice_builder
    
    output = slice_layers[-1].get_output(0)
    slice_network.mark_output(output)
    output.dtype = trt.float16
    output.allowed_formats = 1 << int(trt.TensorFormat.CHW16) | 1 << int(trt.TensorFormat.DLA_LINEAR)
    BUILD_LOG.write(f"slice output: {output.name}, shape: {output.shape}\n")
    BUILD_LOG.flush()

    slice_config.set_flag(trt.BuilderFlag.FP16)
    slice_config.set_flag(trt.BuilderFlag.DIRECT_IO)
    slice_config.DLA_core = 0
    slice_config.default_device_type = trt.DeviceType.DLA
    slice_config.engine_capability = trt.EngineCapability.DLA_STANDALONE
    slice_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    slice_builder.max_batch_size = BATCH_SIZE
    slice_engine = slice_builder.build_serialized_network(slice_network, slice_config)
    assert slice_engine
    with open(f"{ENGINE_DIR}/{slice_idx}.engine", "wb") as f:
        f.write(slice_engine)
    slice_idx += 1

    del slice_engine
    del slice_network
    del slice_config
    del slice_builder
    slice_layers = []
    slice_tensors = {}
    slice_config = None
    slice_network = None
    slice_builder = None


def add_layer(network, inputs, add_func, process_layer=lambda layer: None):
    layer = add_func(network, inputs)
    assert layer
    process_layer(layer)

    global slice_idx
    global slice_layers
    global slice_tensors
    global slice_config
    global slice_network
    global slice_builder

    if slice_builder == None:
        BUILD_LOG.write(f"\ncreate new slice {slice_idx}\n")
        slice_builder = trt.Builder(TRT_LOGGER)
        slice_config = slice_builder.create_builder_config()
        slice_network = slice_builder.create_network()
    
    BUILD_LOG.write(f"adding layer {layer.name} to slice {slice_idx}\n")
    
    slice_inputs = []
    for input in inputs:
        if input.name in slice_tensors:
            slice_input = slice_tensors[input.name]
        else:
            BUILD_LOG.write(f"add input: {input.name}, shape: {input.shape}\n")
            slice_input = slice_network.add_input(input.name, trt.float16, input.shape)
            slice_input.dtype = trt.float16
            slice_input.allowed_formats = 1 << int(trt.TensorFormat.CHW16) | 1 << int(trt.TensorFormat.DLA_LINEAR)
            slice_tensors[input.name] = slice_input
        
        slice_inputs.append(slice_input)
    
    slice_layer = add_func(slice_network, slice_inputs)
    process_layer(slice_layer)
    slice_layer.name = layer.name
    slice_layers.append(slice_layer)

    slice_output = slice_layer.get_output(0)
    slice_output.name = layer.get_output(0).name
    slice_tensors[layer.get_output(0).name] = slice_output

    if slice_config.can_run_on_DLA(slice_layer):
        print(f"Layer {slice_layer.name} can run on DLA.")
    else:
        print(f"Layer {slice_layer.name} CANNOT run on DLA.")

    return layer


def load_weights(file):
    print(f"Loading weights: {file}")

    assert os.path.exists(file), 'Unable to load weight file.'

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert count == len(lines) - 1
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert cur_count + 2 == len(splits)
        values = []
        for j in range(2, len(splits)):
            # hex string to bytes to float
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)

    return weight_map


def add_batch_norm_2d(network, weight_map, input, layer_name, eps):
    gamma = weight_map[layer_name + ".weight"]
    beta = weight_map[layer_name + ".bias"]
    mean = weight_map[layer_name + ".running_mean"]
    var = weight_map[layer_name + ".running_var"]
    var = np.sqrt(var + eps)

    scale = gamma / var
    shift = -mean / var * gamma + beta
    scale = add_layer(network, [input],
        lambda net, inputs: net.add_scale(input=inputs[0],
                                          mode=trt.ScaleMode.CHANNEL,
                                          shift=shift,
                                          scale=scale))
    return scale


def set_conv(layer, s, p, g):
    layer.stride_nd = (s, s)
    layer.padding_nd = (p, p)
    layer.num_groups = g


def conv_bn_relu(network, weight_map, input, outch, ksize, s, g, lname):
    p = (ksize - 1) // 2

    conv1 = add_layer(network, [input],
        lambda net, inputs: net.add_convolution_nd(input=inputs[0],
                                                   num_output_maps=outch,
                                                   kernel_shape=(ksize, ksize),
                                                   kernel=weight_map[lname + "0.weight"],
                                                   bias=trt.Weights()),
        lambda layer: set_conv(layer, s, p, g))

    bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "1", EPS)
    assert bn1

    relu1 = add_layer(network, [bn1.get_output(0)],
        lambda net, inputs: net.add_activation(inputs[0], type=trt.ActivationType.RELU))

    shift = np.array(-6.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    power = np.array(1.0, dtype=np.float32)
    scale1 = add_layer(network, [bn1.get_output(0)],
        lambda net, inputs: net.add_scale(input=inputs[0],
                                          mode=trt.ScaleMode.UNIFORM,
                                          shift=shift,
                                          scale=scale,
                                          power=power))
    relu2 = add_layer(network, [scale1.get_output(0)],
        lambda net, inputs: net.add_activation(inputs[0], type=trt.ActivationType.RELU))
    ew1 = add_layer(network, [relu1.get_output(0), relu2.get_output(0)],
        lambda net, inputs: net.add_elementwise(inputs[0], inputs[1], trt.ElementWiseOperation.SUB))

    return ew1


def inverted_res(network, weight_map, input, lname, inch, outch, s, exp):
    hidden = inch * exp
    use_res_connect = (s == 1 and inch == outch)

    if exp != 1:
        ew1 = conv_bn_relu(network, weight_map, input, hidden, 1, 1, 1, lname + "conv.0.")
        ew2 = conv_bn_relu(network, weight_map, ew1.get_output(0), hidden, 3, s, hidden, lname + "conv.1.")
        conv1 = add_layer(network, [ew2.get_output(0)],
            lambda net, inputs: net.add_convolution(input=inputs[0],
                                                    num_output_maps=outch,
                                                    kernel_shape=(1, 1),
                                                    kernel=weight_map[lname + "conv.2.weight"],
                                                    bias=trt.Weights()))
        bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "conv.3", EPS)
    else:
        ew1 = conv_bn_relu(network, weight_map, input, hidden, 3, s, hidden, lname + "conv.0.")
        conv1 = add_layer(network, [ew1.get_output(0)],
            lambda net, inputs: net.add_convolution(input=inputs[0],
                                                    num_output_maps=outch,
                                                    kernel_shape=(1, 1),
                                                    kernel=weight_map[lname + "conv.1.weight"],
                                                    bias=trt.Weights()))
        bn1 = add_batch_norm_2d(network, weight_map, conv1.get_output(0), lname + "conv.2", EPS)

    if not use_res_connect:
        return bn1

    ew3 = add_layer(network, [input, bn1.get_output(0)],
        lambda net, inputs: net.add_elementwise(inputs[0], inputs[1], trt.ElementWiseOperation.SUM))

    return ew3


def create_engine(max_batch_size, builder, config):
    weight_map = load_weights(WEIGHT_PATH)
    network = builder.create_network()

    data = network.add_input(INPUT_BLOB_NAME, trt.float16, (3, INPUT_H, INPUT_W))
    assert data
    data.allowed_formats = 1 << int(trt.TensorFormat.CHW16)

    ew1 = conv_bn_relu(network, weight_map, data, 32, 3, 2, 1, "features.0.")
    ir1 = inverted_res(network, weight_map, ew1.get_output(0), "features.1.", 32, 16, 1, 1)
    add_slice()
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.2.", 16, 24, 2, 6)
    add_slice()
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.3.", 24, 24, 1, 6)
    add_slice()
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.4.", 24, 32, 2, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.5.", 32, 32, 1, 6)
    add_slice()
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.6.", 32, 32, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.7.", 32, 64, 2, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.8.", 64, 64, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.9.", 64, 64, 1, 6)
    add_slice()
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.10.", 64, 64, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.11.", 64, 96, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.12.", 96, 96, 1, 6)
    add_slice()
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.13.", 96, 96, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.14.", 96, 160, 2, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.15.", 160, 160, 1, 6)
    add_slice()
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.16.", 160, 160, 1, 6)
    ir1 = inverted_res(network, weight_map, ir1.get_output(0), "features.17.", 160, 320, 1, 6)
    ew2 = conv_bn_relu(network, weight_map, ir1.get_output(0), 1280, 1, 1, 1, "features.18.")
 
    pool1 = add_layer(network, [ew2.get_output(0)],
        lambda net, inputs: net.add_pooling(input=inputs[0],
                                            type=trt.PoolingType.AVERAGE,
                                            window_size=trt.DimsHW(7, 7)))
    fc1 = add_layer(network, [pool1.get_output(0)],
        lambda net, inputs: net.add_fully_connected(input=inputs[0],
                                                    num_outputs=OUTPUT_SIZE,
                                                    kernel=weight_map["classifier.1.weight"],
                                                    bias=weight_map["classifier.1.bias"]))
    add_slice()

    output = fc1.get_output(0)
    network.mark_output(output)
    output.name = OUTPUT_BLOB_NAME
    output.dtype = trt.float16
    output.allowed_formats = 1 << int(trt.TensorFormat.CHW16) | 1 << int(trt.TensorFormat.DLA_LINEAR)

    # for int8
    # config.int8_calibrator = RndInt8Calibrator()
    # config.set_flag(trt.BuilderFlag.INT8)

    # for fp16
    config.set_flag(trt.BuilderFlag.FP16)

    config.DLA_core = 0
    config.default_device_type = trt.DeviceType.DLA
    config.engine_capability = trt.EngineCapability.DLA_STANDALONE
    # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.DIRECT_IO)
    # config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 34)
    # Build Engine
    builder.max_batch_size = max_batch_size
    engine = builder.build_serialized_network(network, config)

    del network
    del weight_map

    return engine


def API_to_model(max_batch_size):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    serialized_engine = create_engine(max_batch_size, builder, config)
    assert serialized_engine
    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized_engine)

    del serialized_engine
    del builder
    del config


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action='store_true')
    parser.add_argument("-d", action='store_true')
    args = parser.parse_args()

    if not (args.s ^ args.d):
        print(
            "arguments not right!\n"
            "python mobilenet_v2.py -s   # serialize model to plan file\n"
            "python mobilenet_v2.py -d   # deserialize plan file and run inference"
        )
        sys.exit()

    if args.s:
        API_to_model(BATCH_SIZE)
    else:
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        context = engine.create_execution_context()
        assert context

        data = np.ones((BATCH_SIZE * 3 * INPUT_H * INPUT_W), dtype=np.float32)
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = data

        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        print(f'Output: \n{trt_outputs[0][:10]}\n{trt_outputs[0][-10:]}')
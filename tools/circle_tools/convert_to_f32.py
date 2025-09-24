#!/usr/bin/env python3
import flatbuffers
import sys
import os

# Add the current working directory to sys.path to find the circle package
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from circle.Model import Model
from circle.Tensor import Tensor, TensorT
from circle.Buffer import Buffer, BufferT
from circle.SubGraph import SubGraph, SubGraphT
from circle.Operator import Operator, OperatorT
from circle.TensorType import TensorType
from circle.BuiltinOperator import BuiltinOperator
from circle.QuantizationParameters import QuantizationParameters, QuantizationParametersT
from circle.QuantizationDetails import QuantizationDetails
from circle.TRIXQuantization import TRIXQuantization, TRIXQuantizationT
from extract_offset import extract_offsets

# Import ModelT from the local circle directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'circle'))
from Model import ModelT


def insert_input_oparam_into_weight(operatorT, subgraphT, offset_result):
    print(f'Filling TRIXQuantization in_ch_stride and offset[]')

    # Get the weight tensor (second input)
    weight_tensor_idx = operatorT.inputs[1]
    if weight_tensor_idx == -1:
        print(f"  Warning: Weight tensor is optional (-1), skipping")
        return

    weight_tensorT = subgraphT.tensors[weight_tensor_idx]

    # Update existing quantization parameters
    weight_quant = weight_tensorT.quantization
    trix_quant = weight_quant.details

    # Get the out tensor (3rd input)
    tensor_idx = operatorT.outputs[0]
    if tensor_idx == -1:
        print(f"  Warning: output tensor is optional (-1), skipping")
        return

    tensorT = subgraphT.tensors[tensor_idx]
    tensor_name = tensorT.name
    tensor_name = str(tensor_name)[2:-1] # remove b'' from name
    print(f'tensor name is : {tensor_name}')

    for k,v in offset_result["inch_stride_result"].items():
        print(f'key is {k}')
        if str(k).find(str(tensor_name)) != -1: # matching tensor name
            print(f'found matching tensor {tensor_name}')
            trix_quant.inChStride = v
        else:
            print(f'not matched to {k}')



def insert_input_qparam_into_weight(operatorT, subgraphT):
    # Get the first input tensor (activation tensor)
    input_tensor_idx = operatorT.inputs[0]
    if input_tensor_idx == -1:
        print(f"  Warning: First input is optional (-1), skipping")
        return

    input_tensorT = subgraphT.tensors[input_tensor_idx]

    # Get the weight tensor (second input)
    weight_tensor_idx = operatorT.inputs[1]
    if weight_tensor_idx == -1:
        print(f"  Warning: Weight tensor is optional (-1), skipping")
        return

    weight_tensorT = subgraphT.tensors[weight_tensor_idx]

    # Check if input tensor has quantization parameters
    if input_tensorT.quantization is None:
        print(f"  Warning: Input tensor has no quantization parameters, skipping")
        return

    # Extract quantization parameters from input tensor
    input_quant = input_tensorT.quantization

    # Get scale and zero point from input tensor
    if (input_quant.scale is None or len(input_quant.scale) == 0 or
        input_quant.zeroPoint is None or len(input_quant.zeroPoint) == 0):
        print(f"  Warning: Input tensor has no scale or zero point, skipping")
        return

    input_scale = input_quant.scale[0]  # Use the first scale value
    input_zero_point = input_quant.zeroPoint[0]  # Use the first zero point value

    print(f"  Input tensor quantization: scale={input_scale}, zero_point={input_zero_point}")

    # Create TRIXQuantization object
    trix_quant = TRIXQuantizationT()
    trix_quant.inputScale = float(input_scale)
    trix_quant.inputZp = int(input_zero_point)
    trix_quant.inChStride = 0  # Default value, may need to be calculated based on tensor shape
    trix_quant.offset = None  # Not used in this case

    # Update weight tensor's quantization parameters
    if weight_tensorT.quantization is None:
        # Create new quantization parameters for weight tensor
        weight_quant = QuantizationParametersT()
        weight_quant.detailsType = QuantizationDetails.TRIXQuantization
        weight_quant.details = trix_quant
        weight_quant.quantizedDimension = 0  # Default value
        weight_tensorT.quantization = weight_quant
    else:
        # Update existing quantization parameters
        weight_quant = weight_tensorT.quantization
        weight_quant.detailsType = QuantizationDetails.TRIXQuantization
        weight_quant.details = trix_quant

    print(f"  Updated weight tensor with TRIXQuantization")


def convert_operator_io_to_f32(input_model_path, output_model_path, yml_path):

    # Let's extract offset and input ch tiling size from yaml(internal/encoded_weight_alloc_info.yml)
    offset_result = extract_offsets(yml_path)
    print(f'offset_result: \n{offset_result["offset_result"]}')
    print(f'inch_stride_result: \n{offset_result["inch_stride_result"]}')


    # Load the model
    with open(input_model_path, 'rb') as f:
        buf = f.read()
    
    model = Model.GetRootAs(buf, 0)
    modelT = ModelT.InitFromObj(model)
    
    # Create a new builder to modify the model
    builder = flatbuffers.Builder(1024)
    
    # Iterate through each subgraph
    for i in range(model.SubgraphsLength()):
        subgraphT = modelT.subgraphs[i]
        
        # Iterate through each operator in the subgraph
        olength = len(subgraphT.operators)
        j=0
        weight_key = ""
        while j < len(subgraphT.operators):
            operatorT = subgraphT.operators[j]
            opcode_index = operatorT.opcodeIndex
            operator_code = model.OperatorCodes(opcode_index)
            operatorT = subgraphT.operators[j]


            # Process output tensors because of weight removal
            for k in range(len(operatorT.outputs)):
                tensor_idx = operatorT.outputs[k]
                tensorT = subgraphT.tensors[tensor_idx]
                tensorT.type = TensorType.FLOAT32
                # TODO: remove quant info from F32 Tensor
                if operator_code.BuiltinCode() == BuiltinOperator.FULLY_CONNECTED:
                    print(f' FC layer output tensor name: {str(tensorT.name)}')
                    weight_key = tensorT.name
                   

            if operator_code.BuiltinCode() == BuiltinOperator.QUANTIZE :
               quant_output_index = operatorT.outputs[0]
               FCT = subgraphT.operators[j-1]
               FCT.outputs = [ quant_output_index ]
               del subgraphT.operators[j]
               if len(subgraphT.operators) == j: # j was the last op, thus stop iteration
                   print(f'{j} is the last op')
                   break
               #j = j + 1
               continue



            # Process input tensors
            for k in range(len(operatorT.inputs)):
                tensor_idx = operatorT.inputs[k]
                if tensor_idx != -1:  # Skip optional inputs
                    tensorT = subgraphT.tensors[tensor_idx]
                    buffer_idx = tensorT.buffer
                    if not "weight" in str(tensorT.name) :
                        if type(modelT.buffers[buffer_idx].data) == None: # NonConst Buffer
                            tensorT.type = TensorType.FLOAT32
                            # TODO: remove quant info from F32 Tensor
                    else :
                        if operator_code.BuiltinCode() == BuiltinOperator.FULLY_CONNECTED:
                            modelT.buffers[buffer_idx] = BufferT() # FIXME: Is this valid to purge buffer
                                                                   # It works anyway.
                            tensorT.type = TensorType.TRIX_W4A8
                            print(f"  Weight tensor {tensorT.name} is in processing")
                            # step1. input quant param
                            insert_input_qparam_into_weight(operatorT, subgraphT)
                            # step2. offset parameter + input ch tiling stride
                            insert_input_oparam_into_weight(operatorT, subgraphT, offset_result)

            j = j + 1 # normal index update routine

    builder = flatbuffers.Builder(0)
    builder.Finish(modelT.Pack(builder), "CIR0".encode())

    # write new model to output file
    with open(output_model_path, 'wb') as f:
        f.write(builder.Output())  # For now just save original buffer

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python convert_to_f32.py <input.circle> <output.circle> <yml_file_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    yml_path = sys.argv[3]
    convert_operator_io_to_f32(input_path, output_path, yml_path)

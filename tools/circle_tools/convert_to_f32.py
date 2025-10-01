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

# Import BeautifulLogger
from beautiful_logger import BeautifulLogger

# Initialize logger instance
logger = BeautifulLogger()


def insert_input_oparam_into_weight(operatorT, subgraphT, offset_result):
    logger.process("Filling TRIXQuantization in_ch_stride and offset[]")

    # Get the weight tensor (second input)
    weight_tensor_idx = operatorT.inputs[1]
    if weight_tensor_idx == -1:
        logger.warning("Weight tensor is optional (-1), skipping")
        return

    weight_tensorT = subgraphT.tensors[weight_tensor_idx]

    # Update existing quantization parameters
    weight_quant = weight_tensorT.quantization
    trix_quant = weight_quant.details

    # Get the out tensor (3rd input)
    tensor_idx = operatorT.outputs[0]
    if tensor_idx == -1:
        logger.warning("Output tensor is optional (-1), skipping")
        return

    tensorT = subgraphT.tensors[tensor_idx]
    tensor_name = tensorT.name
    tensor_name = str(tensor_name)[2:-1] # remove b'' from name
    logger.tensor(f"Processing tensor name: {tensor_name}, idx: {tensor_idx}")

    logger.process("Generate TRIXQuantization\'s in_ch_stride")
    for k,v in offset_result["inch_stride_result"].items():
        logger.debug(f"Checking key: {k}")
        if str(k).find(str(tensor_name)) != -1: # matching tensor name
            logger.success(f"Found matching tensor with key: {k}")
            trix_quant.inChStride = v

    logger.process("Generate TRIXQuantization\'s offset")
    for k,v in offset_result["offset_result"].items():
        logger.debug(f"Checking key: {k}")
        if str(k).find(str(tensor_name)) != -1: # matching tensor name
            logger.success(f"Found matching tensor with key: {k}")
            trix_quant.offset = v

def insert_input_qparam_into_weight(operatorT, subgraphT):
    logger.process("Inserting input quantization parameters into weight tensor")
    
    # Get the first input tensor (activation tensor)
    input_tensor_idx = operatorT.inputs[0]
    if input_tensor_idx == -1:
        logger.warning("First input is optional (-1), skipping")
        return

    input_tensorT = subgraphT.tensors[input_tensor_idx]

    # Get the weight tensor (second input)
    weight_tensor_idx = operatorT.inputs[1]
    if weight_tensor_idx == -1:
        logger.warning("Weight tensor is optional (-1), skipping")
        return

    weight_tensorT = subgraphT.tensors[weight_tensor_idx]

    # Check if input tensor has quantization parameters
    if input_tensorT.quantization is None:
        logger.warning("Input tensor has no quantization parameters, skipping")
        return

    # Extract quantization parameters from input tensor
    input_quant = input_tensorT.quantization

    # Get scale and zero point from input tensor
    if (input_quant.scale is None or len(input_quant.scale) == 0 or
        input_quant.zeroPoint is None or len(input_quant.zeroPoint) == 0):
        logger.warning("Input tensor has no scale or zero point, skipping")
        return

    input_scale = input_quant.scale[0]  # Use the first scale value
    input_zero_point = input_quant.zeroPoint[0]  # Use the first zero point value

    logger.debug(f"Input tensor quantization: scale={input_scale}, zero_point={input_zero_point}")

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

    logger.success("Updated weight tensor with TRIXQuantization")


def convert_operator_io_to_f32(input_model_path, output_model_path, yml_path):
    logger.start(f"Starting model conversion: {input_model_path} -> {output_model_path}")
    
    # Let's extract offset and input ch tiling size from yaml(internal/encoded_weight_alloc_info.yml)
    logger.process("Extracting offset and input channel tiling information from YAML")
    offset_result = extract_offsets(yml_path)
    
    # Display offset results in a beautiful box
    logger.box("Offset Results", {k: (v[:10] if isinstance(v, (list, tuple)) else v) for k, v in list(offset_result["offset_result"].items())[:10]} if isinstance(offset_result["offset_result"], dict) else offset_result["offset_result"])
    logger.box("Input Channel Stride Results", {k: (v[:10] if isinstance(v, (list, tuple)) else v) for k, v in list(offset_result["inch_stride_result"].items())[:10]} if isinstance(offset_result["inch_stride_result"], dict) else offset_result["inch_stride_result"])

    # Load the model
    logger.info(f"Loading model from: {input_model_path}")
    with open(input_model_path, 'rb') as f:
        buf = f.read()
    
    model = Model.GetRootAs(buf, 0)
    modelT = ModelT.InitFromObj(model)
    logger.success("Model loaded successfully")
    
    # Create a new builder to modify the model
    builder = flatbuffers.Builder(1024)
    
    # Iterate through each subgraph
    total_subgraphs = model.SubgraphsLength()
    logger.info(f"Processing {total_subgraphs} subgraph(s)")
    
    for i in range(total_subgraphs):
        logger.info(f"Processing subgraph {i+1}/{total_subgraphs}")
        subgraphT = modelT.subgraphs[i]
        
        # Iterate through each operator in the subgraph
        total_operators = len(subgraphT.operators)
        logger.info(f"Processing {total_operators} operator(s) in subgraph {i+1}")
        
        j=0
        weight_key = ""
        while j < len(subgraphT.operators):
            # Show progress for operators
            logger.progress(j+1, total_operators, f"Subgraph {i+1} Operators")
            
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
                    logger.tensor(f"FC layer output tensor name: {str(tensorT.name)}")
                    weight_key = tensorT.name

            if operator_code.BuiltinCode() == BuiltinOperator.QUANTIZE :
               logger.operator("Processing QUANTIZE operator - removing and connecting previous operator")
               quant_output_index = operatorT.outputs[0]
               FCT = subgraphT.operators[j-1]
               FCT.outputs = [ quant_output_index ]
               del subgraphT.operators[j]
               if len(subgraphT.operators) == j: # j was the last op, thus stop iteration
                   logger.debug(f"Operator {j} was the last operator, stopping iteration")
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
                            logger.weight(f"Processing weight tensor: {tensorT.name}")
                            modelT.buffers[buffer_idx] = BufferT() # FIXME: Is this valid to purge buffer
                                                                   # It works anyway.
                            # TODO: this logic cover all TensorType ?
                            input_idx = operatorT.inputs[0]
                            input_tensorT = subgraphT.tensors[input_idx]
                            logger.debug(f"input type is {input_tensorT.type} and weight type is {tensorT.type}")
                            if tensorT.type == TensorType.UINT4:
                                tensorT.type = TensorType.TRIX_W4A8
                            elif input_tensorT.type == TensorType.INT16:
                                tensorT.type = TensorType.TRIX_W8A16
                            elif input_tensorT.type == TensorType.UINT8 and tensorT.type == TensorType.UINT8:
                                tensorT.type = TensorType.TRIX_W8A8
                            else: # no support
                                logger.error("We exit since not supported tensor type")
                                exit(-1);


                            # step1. input quant param
                            insert_input_qparam_into_weight(operatorT, subgraphT)
                            # step2. offset parameter + input ch tiling stride
                            insert_input_oparam_into_weight(operatorT, subgraphT, offset_result)

            j = j + 1 # normal index update routine
        
        logger.success(f"Completed processing subgraph {i+1}")

    logger.info("Building final model")
    builder = flatbuffers.Builder(0)
    builder.Finish(modelT.Pack(builder), "CIR0".encode())

    # write new model to output file
    logger.info(f"Writing converted model to: {output_model_path}")
    with open(output_model_path, 'wb') as f:
        f.write(builder.Output())  # For now just save original buffer
    
    logger.complete(f"Model conversion completed successfully: {output_model_path}")

if __name__ == "__main__":
    import sys

    # Support optional '--no-debug' flag to suppress debug messages
    args = sys.argv[1:]  # exclude script name
    if "--no-debug" in args:
        logger.set_debug(False)
        args.remove("--no-debug")

    if len(args) != 3:
        logger.error("Invalid number of arguments")
        logger.info("Usage: python convert_to_f32.py [--no-debug] <input.circle> <output.circle> <yml_file_path>")
        sys.exit(1)
    
    input_path, output_path, yml_path = args[0], args[1], args[2]
    
    logger.separator()
    logger.start("CIRCLE Model Converter - F32 Conversion Tool")
    logger.separator()
    
    try:
        convert_operator_io_to_f32(input_path, output_path, yml_path)
    except Exception as e:
        logger.error(f"Model conversion failed: {str(e)}")
        sys.exit(1)
    
    logger.separator()
    logger.complete("All operations completed successfully!")
    logger.separator()

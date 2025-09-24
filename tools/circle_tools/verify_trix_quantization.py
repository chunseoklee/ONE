#!/usr/bin/env python3
import flatbuffers
import sys
import os
from beautiful_logger import BeautifulLogger

# Add the circle tools path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'circle'))
logger = BeautifulLogger()

from circle.Model import Model, ModelT
from circle.Tensor import Tensor, TensorT
from circle.Buffer import Buffer, BufferT
from circle.SubGraph import SubGraph, SubGraphT
from circle.Operator import Operator, OperatorT
from circle.OperatorCode import OperatorCode, OperatorCodeT
from circle.BuiltinOperator import BuiltinOperator
from circle.QuantizationParameters import QuantizationParameters, QuantizationParametersT
from circle.QuantizationDetails import QuantizationDetails
from circle.TRIXQuantization import TRIXQuantization, TRIXQuantizationT

def verify_trix_quantization(model_path):
    """
    Verify that TRIXQuantization has been properly added to FullyConnected weight tensors.
    """
    # Load the model
    with open(model_path, 'rb') as f:
        buf = f.read()
    
    model = Model.GetRootAs(buf, 0)
    modelT = ModelT.InitFromObj(model)
    
    logger.info(f"Verifying TRIXQuantization in model: {model_path}")
    logger.separator()
    
    # Iterate through each subgraph
    for i in range(model.SubgraphsLength()):
        subgraphT = modelT.subgraphs[i]
        logger.info(f"\nSubgraph {i}:")
        
        # Iterate through each operator in the subgraph
        for j in range(len(subgraphT.operators)):
            operatorT = subgraphT.operators[j]
            opcode_index = operatorT.opcodeIndex
            operator_code = model.OperatorCodes(opcode_index)
            
            # Check if this is a FullyConnected operation
            if operator_code.BuiltinCode() == BuiltinOperator.FULLY_CONNECTED:
                logger.info(f"\n  FullyConnected operation at index {j}:")
                
                # Get the inputs (typically: [input_tensor, weight_tensor, bias_tensor])
                if len(operatorT.inputs) < 2:
                    logger.warning(f"FullyConnected operation has only {len(operatorT.inputs)} inputs")
                    continue
                
                # Get the first input tensor (activation tensor)
                input_tensor_idx = operatorT.inputs[0]
                if input_tensor_idx != -1:
                    input_tensorT = subgraphT.tensors[input_tensor_idx]
                    logger.info(f"    Input tensor index: {input_tensor_idx}")
                    logger.info(f"    Input tensor name: {input_tensorT.name}")
                    logger.info(f"    Input tensor type: {input_tensorT.type}")
                    
                    if input_tensorT.quantization:
                        input_quant = input_tensorT.quantization
                        if input_quant.scale and len(input_quant.scale) > 0:
                            logger.info(f"    Input tensor scale: {input_quant.scale[0]}")
                        if input_quant.zeroPoint and len(input_quant.zeroPoint) > 0:
                            logger.info(f"    Input tensor zero_point: {input_quant.zeroPoint[0]}")
                
                # Get the weight tensor (second input)
                weight_tensor_idx = operatorT.inputs[1]
                if weight_tensor_idx != -1:
                    weight_tensorT = subgraphT.tensors[weight_tensor_idx]
                    logger.info(f"    Weight tensor index: {weight_tensor_idx}")
                    logger.info(f"    Weight tensor name: {weight_tensorT.name}")
                    logger.info(f"    Weight tensor type: {weight_tensorT.type}")
                    
                    if weight_tensorT.quantization:
                        weight_quant = weight_tensorT.quantization
                        logger.info(f"    Weight tensor has quantization parameters")
                        
                        if weight_quant.detailsType == QuantizationDetails.TRIXQuantization:
                            logger.success(f"    ✓ Weight tensor has TRIXQuantization!")
                            
                            if weight_quant.details:
                                trix_quant = weight_quant.details
                                logger.info(f"      TRIXQuantization inputScale: {trix_quant.inputScale}")
                                logger.info(f"      TRIXQuantization inputZp: {trix_quant.inputZp}")
                                logger.info(f"      TRIXQuantization inChStride: {trix_quant.inChStride}")
                                if trix_quant.offset.any():
                                    logger.info(f"      TRIXQuantization offset: {trix_quant.offset}")
                        else:
                            logger.error(f"    ✗ Weight tensor does not have TRIXQuantization (detailsType: {weight_quant.detailsType})")
                    else:
                        logger.error(f"    ✗ Weight tensor has no quantization parameters")
                # Get the bias tensor (third input) if present
                if len(operatorT.inputs) > 2:
                    bias_tensor_idx = operatorT.inputs[2]
                    if bias_tensor_idx != -1:
                        bias_tensorT = subgraphT.tensors[bias_tensor_idx]
                        logger.info(f"    Bias tensor index: {bias_tensor_idx}")
                        logger.info(f"    Bias tensor name: {bias_tensorT.name}")
                        logger.info(f"    Bias tensor type: {bias_tensorT.type()}")
                        if bias_tensorT.quantization:
                            bias_quant = bias_tensorT.quantization
                            logger.info(f"    Bias tensor has quantization parameters")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python verify_trix_quantization.py <input.circle>")
        logger.info("This script verifies that TRIXQuantization has been properly added")
        logger.info("to weight tensors of FullyConnected operations.")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        logger.error(f"Error: Model file {model_path} does not exist")
        sys.exit(1)
    
    verify_trix_quantization(model_path)

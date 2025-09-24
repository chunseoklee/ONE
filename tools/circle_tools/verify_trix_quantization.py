#!/usr/bin/env python3
import flatbuffers
import sys
import os

# Add the circle tools path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'circle'))

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
    
    print(f"Verifying TRIXQuantization in model: {model_path}")
    print("=" * 50)
    
    # Iterate through each subgraph
    for i in range(model.SubgraphsLength()):
        subgraphT = modelT.subgraphs[i]
        print(f"\nSubgraph {i}:")
        
        # Iterate through each operator in the subgraph
        for j in range(len(subgraphT.operators)):
            operatorT = subgraphT.operators[j]
            opcode_index = operatorT.opcodeIndex
            operator_code = model.OperatorCodes(opcode_index)
            
            # Check if this is a FullyConnected operation
            if operator_code.BuiltinCode() == BuiltinOperator.FULLY_CONNECTED:
                print(f"\n  FullyConnected operation at index {j}:")
                
                # Get the inputs (typically: [input_tensor, weight_tensor, bias_tensor])
                if len(operatorT.inputs) < 2:
                    print(f"    Warning: FullyConnected operation has only {len(operatorT.inputs)} inputs")
                    continue
                
                # Get the first input tensor (activation tensor)
                input_tensor_idx = operatorT.inputs[0]
                if input_tensor_idx != -1:
                    input_tensorT = subgraphT.tensors[input_tensor_idx]
                    print(f"    Input tensor index: {input_tensor_idx}")
                    print(f"    Input tensor name: {input_tensorT.name}")
                    
                    if input_tensorT.quantization:
                        input_quant = input_tensorT.quantization
                        if input_quant.scale and len(input_quant.scale) > 0:
                            print(f"    Input tensor scale: {input_quant.scale[0]}")
                        if input_quant.zeroPoint and len(input_quant.zeroPoint) > 0:
                            print(f"    Input tensor zero_point: {input_quant.zeroPoint[0]}")
                
                # Get the weight tensor (second input)
                weight_tensor_idx = operatorT.inputs[1]
                if weight_tensor_idx != -1:
                    weight_tensorT = subgraphT.tensors[weight_tensor_idx]
                    print(f"    Weight tensor index: {weight_tensor_idx}")
                    print(f"    Weight tensor name: {weight_tensorT.name}")
                    
                    if weight_tensorT.quantization:
                        weight_quant = weight_tensorT.quantization
                        print(f"    Weight tensor has quantization parameters")
                        
                        if weight_quant.detailsType == QuantizationDetails.TRIXQuantization:
                            print(f"    ✓ Weight tensor has TRIXQuantization!")
                            
                            if weight_quant.details:
                                trix_quant = weight_quant.details
                                print(f"      TRIXQuantization inputScale: {trix_quant.inputScale}")
                                print(f"      TRIXQuantization inputZp: {trix_quant.inputZp}")
                                print(f"      TRIXQuantization inChStride: {trix_quant.inChStride}")
                                if trix_quant.offset.any():
                                    print(f"      TRIXQuantization offset: {trix_quant.offset}")
                        else:
                            print(f"    ✗ Weight tensor does not have TRIXQuantization (detailsType: {weight_quant.detailsType})")
                    else:
                        print(f"    ✗ Weight tensor has no quantization parameters")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_trix_quantization.py <input.circle>")
        print("This script verifies that TRIXQuantization has been properly added")
        print("to weight tensors of FullyConnected operations.")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        sys.exit(1)
    
    verify_trix_quantization(model_path)

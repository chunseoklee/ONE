#!/usr/bin/env python3
import flatbuffers
from circle.Model import Model, ModelT
from circle.Tensor import Tensor, TensorT
from circle.Buffer import Buffer, BufferT
from circle.SubGraph import SubGraph, SubGraphT
from circle.Operator import Operator, OperatorT

def convert_operator_io_to_f32(input_model_path, output_model_path):
    # Load the model
    with open(input_model_path, 'rb') as f:
        buf = f.read()
    
    model = Model.GetRootAs(buf, 0)
    modelT = ModelT.InitFromObj(model)
    
    # Create a new builder to modify the model
    builder = flatbuffers.Builder(1024)
    
    # Iterate through each subgraph
    for i in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(i)
        subgraphT = modelT.subgraphs[i]
        
        # Iterate through each operator in the subgraph
        for j in range(subgraph.OperatorsLength()):
            operator = subgraph.Operators(j)
            operatorT = subgraphT.operators[j]
            
            # Process input tensors
            for k in range(operator.InputsLength()):
                tensor_idx = operator.Inputs(k)
                if tensor_idx != -1:  # Skip optional inputs
                    tensor = subgraph.Tensors(tensor_idx)
                    tensorT = subgraphT.tensors[tensor_idx]
                    if not "weight" in str(tensor.Name()) :
                        tensorT.type = 0 # 0 is FLOAT32
                    else :
                        buffer_idx = tensorT.buffer
                        modelT.buffers[buffer_idx] = BufferT()

            # Process output tensors
            for k in range(operator.OutputsLength()):
                tensor_idx = operator.Outputs(k)

                tensor = subgraph.Tensors(tensor_idx)
                tensorT = subgraphT.tensors[tensor_idx]
                if not "weight" in str(tensor.Name()):
                    tensorT.type = 0


    builder = flatbuffers.Builder(0)
    builder.Finish(modelT.Pack(builder))

    # write new model to output file
    with open(output_model_path, 'wb') as f:
        f.write(builder.Output())  # For now just save original buffer

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_to_f32.py <input.circle> <output.circle>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    convert_operator_io_to_f32(input_path, output_path)

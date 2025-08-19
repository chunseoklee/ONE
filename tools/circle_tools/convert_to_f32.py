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
        subgraphT = modelT.subgraphs[i]
        
        # Iterate through each operator in the subgraph
        olength = len(subgraphT.operators)
        j=0
        while j < olength:
            operatorT = subgraphT.operators[j]
            opcode_index = operatorT.opcodeIndex
            operator_code = model.OperatorCodes(opcode_index)
            operatorT = subgraphT.operators[j]


            # Process output tensors because of weight removal
            for k in range(len(operatorT.outputs)):
                tensor_idx = operatorT.outputs[k]
                tensorT = subgraphT.tensors[tensor_idx]
                tensorT.type = 0



            if operator_code.BuiltinCode() == 114 : # 114 -> QUANTIZE FIXME: FC->QUANTIZE pattern
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
                            tensorT.type = 0 # 0 is FLOAT32
                    else :
                        modelT.buffers[buffer_idx] = BufferT() # FIXME: Is this valid to purge buffer
                                                               # It works anyway.

            j = j + 1 # normal index update routine

    builder = flatbuffers.Builder(0)
    builder.Finish(modelT.Pack(builder), "CIR0".encode())

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

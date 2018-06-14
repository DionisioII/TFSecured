//
//  GraphDefDecryptor.hpp
//  TFSecured
//
//  Created by user on 6/14/18.
//  Copyright © 2018 user. All rights reserved.
//

#include <stdio.h>

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/public/session.h>

#include <tensorflow/core/framework/shape_inference.h>
#include <iostream>
#include <fstream>
#include "picosha2.hpp"
#include "aes.hpp"

using namespace tensorflow;

namespace tf_secured {

    
    const int32_t AES_BLOCK_SIZE       = AES_BLOCKLEN;
    const int32_t AES_INIT_VECTOR_SIZE = AES_BLOCK_SIZE;

    
    inline void GraphDefDecrypt(GraphDef &graph,  const std::array<uint8_t, 32> &keyByteArray);
    inline void GraphDefDecrypt(GraphDef &graph,  const std::string &key256);

    
    
    inline void GraphDefDecrypt(GraphDef &graph,
                                const std::string &key256) {
        std::array<uint8_t, 32> hashKey;
        
        picosha2::hash256_bytes(key256, hashKey);
        GraphDefDecrypt(graph, hashKey);
    }
    
    inline void GraphDefDecrypt(GraphDef &graph,
                                const std::array<uint8_t, 32> &keyByteArray) {
        
        AES_ctx _aesCtx;

        for (NodeDef& node : *graph.mutable_node()) {
            
            if (node.op() != "Const") continue;
            auto attr = node.mutable_attr();
            if (attr->count("value") == 0) continue;
            
            auto mutable_tensor = attr->at("value").mutable_tensor();
            const std::string &tensor_content = mutable_tensor->tensor_content();
            const uint32_t content_size = (uint32_t)mutable_tensor->ByteSizeLong();
            
            
            std::vector<uint8_t> iv_bytes(tensor_content.begin(),
                                          tensor_content.begin() + AES_INIT_VECTOR_SIZE);

            AES_init_ctx_iv(&_aesCtx, keyByteArray.data(), iv_bytes.data());
            
            std::vector<uint8_t> tensor_bytes(tensor_content.begin() + AES_INIT_VECTOR_SIZE,
                                              tensor_content.end());
            
            AES_CBC_decrypt_buffer(&_aesCtx, tensor_bytes.data(), content_size-AES_INIT_VECTOR_SIZE);
            
            const size_t tensor_size = tensor_bytes.size();
            const int last_index = (int)tensor_bytes[tensor_size - 1];
            size_t size_without_padding = tensor_size - last_index;
            mutable_tensor->set_tensor_content(tensor_bytes.data(), size_without_padding);
            
#ifdef DEBUG
            std::cout   << "Node: " << node.name()
                        << ",\n     op: " << node.op()
                        << "\n content size (" << content_size << "): " << tensor_content.size() << "\n";
#endif
            
        }
        // Save Model:
        //    std::fstream file;
        //    file.open("filename");
        //    bool success = graph.SerializeToOstream(&file);
        //    file.close();
    }
}

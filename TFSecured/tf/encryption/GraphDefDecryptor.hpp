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

    typedef enum {
        DecryptOnlyConstants,
        DecryptAll
    } GraphDecryptMode;
    
    inline void GraphDefDecrypt(GraphDef &graph,
                                const std::array<uint8_t, 32> &keyByteArray,
                                const GraphDecryptMode decryptMode = GraphDecryptMode::DecryptOnlyConstants);
    
    inline void GraphDefDecrypt(GraphDef &graph,
                                const std::string &key256,
                                const GraphDecryptMode decryptMode = GraphDecryptMode::DecryptOnlyConstants);
    
    inline void decryptNodeNames(NodeDef& node,
                                 struct AES_ctx *ctx,
                                 const std::array<uint8_t, 32> &keyByteArray);
    
    inline const std::vector<uint8_t> decrypt(struct AES_ctx *ctx,
                                              const std::array<uint8_t, 32> &keyByteArray,
                                              const std::string& input,
                                              const uint32_t content_size) {

        const std::vector<uint8_t> iv_bytes(input.begin(),
                                            input.begin() + AES_INIT_VECTOR_SIZE);
        
        AES_init_ctx_iv(ctx, keyByteArray.data(), iv_bytes.data());
        
        std::vector<uint8_t> tensor_bytes(input.begin() + AES_INIT_VECTOR_SIZE,
                                          input.end());
        
        AES_CBC_decrypt_buffer(ctx, tensor_bytes.data(), content_size-AES_INIT_VECTOR_SIZE);
        
        const size_t tensor_size = tensor_bytes.size();
        const int last_index = (int)tensor_bytes[tensor_size - 1];
        const size_t size_without_padding = tensor_size - last_index;
        tensor_bytes.resize(size_without_padding);
        return tensor_bytes;
    }
    
    inline const std::string decryptToString(struct AES_ctx *ctx,
                                             const std::array<uint8_t, 32> &keyByteArray,
                                             const std::string& tensor_content,
                                             const uint32_t content_size) {
        std::vector<uint8_t> decryptBytes = decrypt(ctx,
                                                    keyByteArray,
                                                    tensor_content,
                                                    content_size);
        return std::string(reinterpret_cast<const char*>(decryptBytes.data()),
                           decryptBytes.size());
    }
    
    
        
    
    inline void GraphDefDecrypt(GraphDef &graph,
                                const std::string &key256,
                                const GraphDecryptMode decryptMode) {
        std::array<uint8_t, 32> hashKey;
        picosha2::hash256_bytes(key256, hashKey);
        GraphDefDecrypt(graph, hashKey);
    }
    
    inline void GraphDefDecrypt(GraphDef &graph,
                                const std::array<uint8_t, 32> &keyByteArray,
                                const GraphDecryptMode decryptMode) {
        
        AES_ctx aesCtx;

        for (NodeDef& node : *graph.mutable_node()) {
            if (decryptMode == GraphDecryptMode::DecryptAll) {
                decryptNodeNames(node, &aesCtx, keyByteArray);
            }
#ifdef DEBUG
            std::cout   << "Node: " << node.name()
                        << ",\n     op: " << node.op() << std::endl;
#endif

            if (node.op() != "Const") continue;
            auto attr = node.mutable_attr();
            if (attr->count("value") == 0) continue;
            
            auto mutable_tensor = attr->at("value").mutable_tensor();
            const std::string &tensor_content = mutable_tensor->tensor_content();
            const uint32_t content_size = (uint32_t)mutable_tensor->ByteSizeLong();
            
            const std::string decrypted_tensor = decryptToString(&aesCtx,
                                                                 keyByteArray,
                                                                 tensor_content,
                                                                 content_size);
            mutable_tensor->set_tensor_content(decrypted_tensor);
        }
        // Save Model:
        //    std::fstream file;
        //    file.open("filename");
        //    bool success = graph.SerializeToOstream(&file);
        //    file.close();
    }
    
    
    inline void decryptNodeNames(NodeDef& node,
                                 struct AES_ctx *ctx,
                                 const std::array<uint8_t, 32> &keyByteArray) {
        const std::string &op = node.op();
        const std::string &name = node.name();
        const std::string &new_name =  decryptToString(ctx,
                                                       keyByteArray,
                                                       name,
                                                       (uint32_t)name.size());
        const std::string &new_op = decryptToString(ctx,
                                                    keyByteArray,
                                                    op,
                                                    (uint32_t)op.size());
        node.set_op(new_op);
        node.set_name(new_name);
    }
}

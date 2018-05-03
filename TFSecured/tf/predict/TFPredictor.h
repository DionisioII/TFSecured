//
//  TFPredictor.h
//  TFSecured
//
//  Created by user on 5/3/18.
//  Copyright © 2018 user. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIImage.h>

#include <tensorflow/core/framework/tensor.h>

NS_ASSUME_NONNULL_BEGIN

typedef void(^TFErrorCallback)(NSError *error);

@interface TFPredictor : NSObject



+ (instancetype)initWith:(NSString*)modelPath
           inputNodeName:(NSString*)inNode
          outputNodeName:(NSString*)outNode;


- (void)loadModel:(nullable TFErrorCallback) callback;

- (void)predictTensor:(const tensorflow::Tensor&)input output: (tensorflow::Tensor*)output;

@end

NS_ASSUME_NONNULL_END

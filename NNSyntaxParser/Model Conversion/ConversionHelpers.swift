//
//  ConversionHelpers.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-31.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow

func nnLayer<T: RawRepresentable>(name: T, inputNames: [T], outputNames: [T], layer: CoreML_Specification_NeuralNetworkLayer.OneOf_Layer, inputDimensions: [[Int]], outputDimensions: [[Int]]) -> CoreML_Specification_NeuralNetworkLayer where T.RawValue == String {
    return nnLayer(name: name.rawValue, inputNames: inputNames.map({ $0.rawValue }), outputNames: outputNames.map({ $0.rawValue }), layer: layer, inputDimensions: inputDimensions, outputDimensions: outputDimensions)
}

func nnLayer(name: String, inputNames: [String], outputNames: [String], layer: CoreML_Specification_NeuralNetworkLayer.OneOf_Layer, inputDimensions: [[Int]], outputDimensions: [[Int]]) -> CoreML_Specification_NeuralNetworkLayer {
    var nnLayer = CoreML_Specification_NeuralNetworkLayer()
    nnLayer.name = name
    nnLayer.input = inputNames
    nnLayer.output = outputNames
    nnLayer.layer = layer
    nnLayer.inputTensor = inputDimensions.map({ layerDimension(dimension: $0) })
    nnLayer.outputTensor = outputDimensions.map({ layerDimension(dimension: $0) })
    return nnLayer
}

/// Makes an InnerProductLayer (aka Dense layer)
/// - Parameters:
///   - numInput: dimension of the input (C_in)
///   - numOutput: dimension of the output (C_out)
///   - weights: weights of the dense layer with shape [C_out, C_in]
///   - bias: bias weights (if any) with dimension (C_out)
func makeInnerProductLayer(numInput: Int, numOutput: Int, weights: Tensor<Float>, bias: Tensor<Float>? = nil) -> CoreML_Specification_InnerProductLayerParams {
    var innerProductLayer = CoreML_Specification_InnerProductLayerParams()
    innerProductLayer.inputChannels = UInt64(numInput)
    innerProductLayer.outputChannels = UInt64(numOutput)
    innerProductLayer.weights = makeWeightParams(weights: weights.flattened())
    if let bias = bias {
        innerProductLayer.hasBias_p = true
        innerProductLayer.bias = makeWeightParams(weights: bias.flattened())
    } else {
        innerProductLayer.hasBias_p = false
    }
    return innerProductLayer
}

func layerDimension(dimension: [Int]) -> CoreML_Specification_Tensor {
    var tensor = CoreML_Specification_Tensor()
    tensor.dimValue = dimension.map({ Int64($0) })
    tensor.rank = UInt32(dimension.count)
    return tensor
}

func makeWeightParams(weights: Tensor<Float>) -> CoreML_Specification_WeightParams {
    var params = CoreML_Specification_WeightParams()
    params.floatValue = weights.scalars
    return params
}

func makeFeatureDescription(name: String, shortDescription: String, type: CoreML_Specification_FeatureType.OneOf_Type) -> CoreML_Specification_FeatureDescription {
    var featureType = CoreML_Specification_FeatureType()
    featureType.type = type
    
    var builder = CoreML_Specification_FeatureDescription()
    builder.name = name
    builder.shortDescription = shortDescription
    builder.type = featureType
    return builder
}

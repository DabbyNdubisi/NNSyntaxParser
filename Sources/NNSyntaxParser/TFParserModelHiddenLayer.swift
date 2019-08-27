//
//  TFParserModelHiddenLayer.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-08-04.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow

struct TFParserModelHiddenLayer: Layer {
    var wordWeights: Tensor<Float> = Tensor<Float>(randomUniform: w_WeightDimension)
    var tagWeights: Tensor<Float> = Tensor<Float>(randomUniform: t_WeightDimension)
    var labelWeights: Tensor<Float> = Tensor<Float>(randomUniform: l_WeightDimension)
    var bias: Tensor<Float> = Tensor<Float>(zeros: bias_Dimension)
    @noDerivative let activation: Dense<Float>.Activation = relu
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let inputs = input.split(sizes: Tensor<Int32>([Int32(w_WeightDimension[0]), Int32(t_WeightDimension[0]), Int32(l_WeightDimension[0])]), alongAxis: 1)
        return activation(
            matmul(inputs[0], wordWeights) +
                matmul(inputs[1], tagWeights) +
                matmul(inputs[2], labelWeights) +
                bias
        )
    }
}

// MARK: Codable
extension TFParserModelHiddenLayer: Codable {
    private enum CodingKeys: String, CodingKey {
        case wordWeights
        case tagWeights
        case labelWeights
        case bias
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        wordWeights = try container.decode(Tensor<Float>.self, forKey: .wordWeights)
        tagWeights = try container.decode(Tensor<Float>.self, forKey: .tagWeights)
        labelWeights = try container.decode(Tensor<Float>.self, forKey: .labelWeights)
        bias = try container.decode(Tensor<Float>.self, forKey: .bias)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(wordWeights, forKey: .wordWeights)
        try container.encode(tagWeights, forKey: .tagWeights)
        try container.encode(labelWeights, forKey: .labelWeights)
        try container.encode(bias, forKey: .bias)
    }
}

// MARK: - Helpers
extension TFParserModelHiddenLayer {
    var l2Loss: Tensor<Float> {
        return Raw.l2Loss(t: wordWeights) +
        Raw.l2Loss(t: tagWeights) +
        Raw.l2Loss(t: labelWeights) +
        Raw.l2Loss(t: bias)
    }
}

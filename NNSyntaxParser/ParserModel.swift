//
//  ParserModel.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-11.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow

// <NONE> + POSTags + Dependency relations
let domainTokensVocabularySize = TransitionFeatureProvider.domainTokens.count
let wordsVocabularySize = 400000
let totalVocabularySize = (domainTokensVocabularySize + wordsVocabularySize)
let numLabels = Transition.numberOfTransitions
let dh: Int = 200 // hidden layer size
let n_wFeatures: Int = 18 // number of word features
let n_tFeatures: Int = 18 // number of tag features
let n_lFeatures: Int = 12 // number of label features
let n_Features: Int = n_wFeatures + n_tFeatures + n_lFeatures
let embeddingDimension: Int = 50 // embedding vector dimension
let w_WeightDimension: TensorShape = [n_wFeatures * embeddingDimension, dh]
let t_WeightDimension: TensorShape = [n_tFeatures * embeddingDimension, dh]
let l_WeightDimension: TensorShape = [n_lFeatures * embeddingDimension, dh]
let bias_Dimension: TensorShape = [dh]

struct ParserModelHiddenLayer: Layer {
    var wordWeights: Tensor<Float> = Tensor<Float>(randomUniform: w_WeightDimension)
    var tagWeights: Tensor<Float> = Tensor<Float>(randomUniform: t_WeightDimension)
    var labelWeights: Tensor<Float> = Tensor<Float>(randomUniform: l_WeightDimension)
    var bias: Tensor<Float> = Tensor<Float>(zeros: bias_Dimension)
    @noDerivative let activation: Dense<Float>.Activation = { $0 * $0 * $0 }
    
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

struct ParserModel: Layer {
    // Need to make sure look up table covers (words + tags + labels)
    var embeddingLayer: Embedding<Float>
    var flattenLayer = Flatten<Float>()
    var parseLayer = ParserModelHiddenLayer()
    var outputLayer = Dense<Float>(inputSize: dh, outputSize: numLabels)
    
    init(embeddings: Tensor<Float>) {
        self.embeddingLayer = Embedding(embeddings: embeddings)
    }
    
    init() {
        self.embeddingLayer = Embedding(vocabularySize: totalVocabularySize, embeddingSize: embeddingDimension)
    }
    
    @differentiable
    func callAsFunction(_ input: EmbeddingInput) -> Tensor<Float> {
//        return input.sequenced(through: embeddingLayer, flattenLayer, parseLayer)
        return input.sequenced(through: embeddingLayer, flattenLayer, parseLayer, outputLayer)
    }
}

// MARK: - ParserModelHiddenLayer + Codable
extension ParserModelHiddenLayer: Codable {
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

// MARK: - ParserModel + Codable
extension ParserModel: Codable {
    private enum CodingKeys: String, CodingKey {
        case embeddingLayer
        case parseLayer
        case outputLayerWeights
        case outputLayerBias
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        let embeddingLayerWeights = try container.decode(Tensor<Float>.self, forKey: .embeddingLayer)
        let outputLayerWeights = try container.decode(Tensor<Float>.self, forKey: .outputLayerWeights)
        let outputLayerBias = try container.decode(Tensor<Float>.self, forKey: .outputLayerBias)
        
        embeddingLayer = Embedding(embeddings: embeddingLayerWeights)
        parseLayer = try container.decode(ParserModelHiddenLayer.self, forKey: .parseLayer)
        outputLayer = Dense<Float>(weight: outputLayerWeights, bias: outputLayerBias, activation: identity)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(embeddingLayer.embeddings, forKey: .embeddingLayer)
        try container.encode(parseLayer, forKey: .parseLayer)
        try container.encode(outputLayer.weight, forKey: .outputLayerWeights)
        try container.encode(outputLayer.bias, forKey: .outputLayerBias)
    }
}

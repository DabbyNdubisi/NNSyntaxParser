//
//  TFParserModel.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-11.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow

struct TFParserModel: Layer {
    // Need to make sure look up table covers (words + tags + labels)
    var embeddingLayer: Embedding<Float>
    var flattenLayer = Flatten<Float>()
    var parseLayer = TFParserModelHiddenLayer()
    var outputLayer = Dense<Float>(inputSize: dh, outputSize: numLabels)
    
    init(embeddings: Tensor<Float>) {
        self.embeddingLayer = Embedding(embeddings: embeddings)
    }
    
    init() {
        self.embeddingLayer = Embedding(vocabularySize: totalVocabularySize, embeddingSize: embeddingDimension)
    }
    
    @differentiable
    func callAsFunction(_ input: EmbeddingInput) -> Tensor<Float> {
        return input.sequenced(through: embeddingLayer, flattenLayer, parseLayer, outputLayer)
    }
}

// MARK: - Codable
extension TFParserModel: Codable {
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
        parseLayer = try container.decode(TFParserModelHiddenLayer.self, forKey: .parseLayer)
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

// MARK: - Helpers
extension TFParserModel {
    var l2Loss: Tensor<Float> {
        return Raw.l2Loss(t: embeddingLayer.embeddings) +
            parseLayer.l2Loss +
            Raw.l2Loss(t: outputLayer.weight) + Raw.l2Loss(t: outputLayer.bias)
    }
}

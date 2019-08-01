//
//  MLParserModelConverter.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-25.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow
import SwiftProtobuf

struct MLParserModelConverter {
    private enum LayerName: String, RawRepresentable {
        case input = "parseState"
        case expand2d = "parse_state_expanded"
        case embedding = "embedding"
        case flatten = "flatten_reshape"
        // Hidden layer 1
        case splitHidden1 = "split_hidden_layer_1"
        case splitHidden1_nw = "split_hidden_layer_1_nw"
        case splitHidden1_nt = "split_hidden_layer_1_nt"
        case splitHidden1_nl = "split_hidden_layer_1_nl"
        case denseHidden1_nw = "nw_dense_hidden_layer_1"
        case denseHidden1_nt = "nt_dense_hidden_layer_1"
        case denseHidden1_nl = "nl_dense_hidden_layer_1"
        case addHidden1_nwtl = "(nw+nt+nl)_add_dense_hidden_layer_1"
        case biasHidden1 = "bias_hidden_layer_1"
        case preActivationReshapeHidden1 = "pre_activation_reshape_hidden_layer_1"
        case activationHidden1 = "cube_activation_hidden_layer_1"
        // Output Layer
        case output = "outputTransitionLogits"
    }
    
    let model: TFParserModel
    
    func convertToMLModel() throws -> Data {
        var modelBuilder = CoreML_Specification_Model()
        
        modelBuilder.specificationVersion = 4
        modelBuilder.description_p = makeDescription()
        modelBuilder.isUpdatable = false
        modelBuilder.neuralNetwork = makeNeuralNetwork()
        return try modelBuilder.serializedData()
    }
    
    // MARK: - NeuralNetworkClassifier
    func makeNeuralNetwork() -> CoreML_Specification_NeuralNetwork {
        var nnBuilder = CoreML_Specification_NeuralNetwork()
        // layers
        var layers = [[CoreML_Specification_NeuralNetworkLayer]]()
        layers.append(makeExpandLayer())
        layers.append(makeEmbeddingLayer())
        layers.append(makeFlattenLayer())
        layers.append(makeCustomHiddenLayer())
        layers.append(makeOutputLayer())
        nnBuilder.layers = layers.flatMap({ $0 })

        nnBuilder.arrayInputShapeMapping = .exactArrayMapping
        return nnBuilder
    }
    
    // MARK: - ModelDescription
    private func makeDescription() -> CoreML_Specification_ModelDescription {
        var builder = CoreML_Specification_ModelDescription()
        builder.input = makeInputFeaturesDescription()
        builder.output = makeOutputFeatureDescription()
        builder.metadata = makeMetadata()
        return builder
    }
    
    private func makeInputFeaturesDescription() -> [CoreML_Specification_FeatureDescription] {
        typealias InputFeature = (name: String, description: String, type: CoreML_Specification_FeatureType.OneOf_Type)
        let inputFeatures: [InputFeature] = [
            (LayerName.input.rawValue,
             "features of the current automata state",
             .multiArrayType({
                var arrayType = CoreML_Specification_ArrayFeatureType()
                arrayType.dataType = .int32
                arrayType.shape = [Int64(n_Features)]
                return arrayType }())
            )
        ]
        
        return inputFeatures.map {
           makeFeatureDescription(name: $0.name, shortDescription: $0.description, type: $0.type)
        }
    }
    
    private func makeOutputFeatureDescription() -> [CoreML_Specification_FeatureDescription] {
        typealias OutputFeature = (name: String, description: String, type: CoreML_Specification_FeatureType.OneOf_Type)
        let outputFeatures: [OutputFeature] = [
            (LayerName.output.rawValue,
             "the logits distribution of all transition labels",
             .multiArrayType({
                var arrayType = CoreML_Specification_ArrayFeatureType()
                arrayType.dataType = .float32
                arrayType.shape = [Int64(numLabels)]
                return arrayType}())
            )
        ]

        return outputFeatures.map {
            makeFeatureDescription(name: $0.name, shortDescription: $0.description, type: $0.type)
        }
    }

    private func makeMetadata() -> CoreML_Specification_Metadata {
        var metadata = CoreML_Specification_Metadata()
        metadata.author = "Dabby Ndubisi"
        metadata.shortDescription = "Classifier for Transition based Syntactic parsing of English language"
        metadata.license = ""
        return metadata
    }
}

// MARK: - Layer conversions
private extension MLParserModelConverter {
    func makeExpandLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        var expandLayer = CoreML_Specification_ExpandDimsLayerParams()
        expandLayer.axes = [1, 2]
        return [nnLayer(
            name: LayerName.expand2d,
            inputNames: [LayerName.input],
            outputNames: [LayerName.expand2d],
            layer: .expandDims(expandLayer),
            inputDimensions: [[n_Features]],
            outputDimensions: [[n_Features, 1, 1]]
        )]
    }
    
    func makeEmbeddingLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        var embeddingLayer = CoreML_Specification_EmbeddingNDLayerParams()
        embeddingLayer.vocabSize = UInt64(totalVocabularySize)
        embeddingLayer.embeddingSize = UInt64(embeddingDimension)
        embeddingLayer.weights = makeWeightParams(weights: model.embeddingLayer.embeddings.transposed())
        
        return [nnLayer(
            name: LayerName.embedding,
            inputNames: [LayerName.expand2d],
            outputNames: [LayerName.embedding],
            layer: .embeddingNd(embeddingLayer),
            inputDimensions: [[n_Features, 1, 1]],
            outputDimensions: [[n_Features, 1, embeddingDimension]]
        )]
    }
    
    func makeFlattenLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        var reshape = CoreML_Specification_ReshapeStaticLayerParams()
        reshape.targetShape = [1, 1, Int64(n_Features * embeddingDimension)]
        
        return [nnLayer(
            name: LayerName.flatten,
            inputNames: [LayerName.embedding],
            outputNames: [LayerName.flatten],
            layer: .reshapeStatic(reshape),
            inputDimensions: [[n_Features, 1, embeddingDimension]],
            outputDimensions: [[1, 1, n_Features * embeddingDimension]]
        )]
    }
    
    func makeCustomHiddenLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        // Split layer
        func makeSplitLayer() -> CoreML_Specification_NeuralNetworkLayer {
            var splitLayer = CoreML_Specification_SplitNDLayerParams()
            splitLayer.axis = -1
            splitLayer.splitSizes = [
                UInt64(n_wFeatures * embeddingDimension),
                UInt64(n_tFeatures * embeddingDimension),
                UInt64(n_lFeatures * embeddingDimension)
            ]
            splitLayer.numSplits = UInt64(splitLayer.splitSizes.count)
            let output = [
                [1, 1, n_wFeatures * embeddingDimension],
                [1, 1, n_tFeatures * embeddingDimension],
                [1, 1, n_lFeatures * embeddingDimension]
            ]
            return nnLayer(
                name: LayerName.splitHidden1,
                inputNames: [LayerName.flatten],
                outputNames: [LayerName.splitHidden1_nw, LayerName.splitHidden1_nt, LayerName.splitHidden1_nl],
                layer: .splitNd(splitLayer),
                inputDimensions: [[1, 1, n_Features * embeddingDimension]],
                outputDimensions: output
            )
        }
        
        // n_w Inner Product layer
        func makeNwDenseLayer() -> CoreML_Specification_NeuralNetworkLayer {
            let innerProductLayer = makeInnerProductLayer(numInput: n_wFeatures * embeddingDimension, numOutput: dh, weights: model.parseLayer.wordWeights.transposed())
            
            return nnLayer(
                name: LayerName.denseHidden1_nw,
                inputNames: [LayerName.splitHidden1_nw],
                outputNames: [LayerName.denseHidden1_nw],
                layer: .innerProduct(innerProductLayer),
                inputDimensions: [[1, 1, n_wFeatures * embeddingDimension]],
                outputDimensions: [[1, 1, dh]]
            )
        }
        
        // n_t InnerProduct layer
        func makeNtDenseLayer() -> CoreML_Specification_NeuralNetworkLayer {
            let innerProductLayer = makeInnerProductLayer(numInput: n_tFeatures * embeddingDimension, numOutput: dh, weights: model.parseLayer.tagWeights.transposed())
            
            return nnLayer(
                name: LayerName.denseHidden1_nt,
                inputNames: [LayerName.splitHidden1_nt],
                outputNames: [LayerName.denseHidden1_nt],
                layer: .innerProduct(innerProductLayer),
                inputDimensions: [[1, 1, n_tFeatures * embeddingDimension]],
                outputDimensions: [[1, 1, dh]]
            )
        }
        
        // n_l InnerProduct layer
        func makeNlDenseLayer() -> CoreML_Specification_NeuralNetworkLayer {
            let innerProductLayer = makeInnerProductLayer(numInput: n_lFeatures * embeddingDimension, numOutput: dh, weights: model.parseLayer.labelWeights.transposed())
            
            return nnLayer(
                name: LayerName.denseHidden1_nl,
                inputNames: [LayerName.splitHidden1_nl],
                outputNames: [LayerName.denseHidden1_nl],
                layer: .innerProduct(innerProductLayer),
                inputDimensions: [[1, 1, n_lFeatures * embeddingDimension]],
                outputDimensions: [[1, 1, dh]]
            )
        }
        
        // Add layer
        func makeAddLayer() -> CoreML_Specification_NeuralNetworkLayer {
            return nnLayer(
                name: LayerName.addHidden1_nwtl,
                inputNames: [LayerName.denseHidden1_nw, LayerName.denseHidden1_nt, LayerName.denseHidden1_nl],
                outputNames: [LayerName.addHidden1_nwtl],
                layer: .add(CoreML_Specification_AddLayerParams()),
                inputDimensions: [[1, 1, dh], [1, 1, dh], [1, 1, dh]],
                outputDimensions: [[1, 1, dh]]
            )
        }
        
        // Bias Layer
        func makeBiasLayer() -> CoreML_Specification_NeuralNetworkLayer {
            var biasLayer = CoreML_Specification_BiasLayerParams()
            biasLayer.shape = [1, 1, UInt64(dh)]
            biasLayer.bias = makeWeightParams(weights: model.parseLayer.bias.transposed())
            
            return nnLayer(
                name: LayerName.biasHidden1,
                inputNames: [LayerName.addHidden1_nwtl],
                outputNames: [LayerName.biasHidden1],
                layer: .bias(biasLayer),
                inputDimensions: [[1, 1, dh]],
                outputDimensions: [[1, 1, dh]]
            )
        }
        
        func makePreActivationReshapeLayer() -> CoreML_Specification_NeuralNetworkLayer {
            var biasReshape = CoreML_Specification_SqueezeLayerParams()
            biasReshape.squeezeAll = true
            
            return nnLayer(
                name: LayerName.preActivationReshapeHidden1,
                inputNames: [LayerName.biasHidden1],
                outputNames: [LayerName.preActivationReshapeHidden1],
                layer: .squeeze(biasReshape),
                inputDimensions: [[1, 1, dh]],
                outputDimensions: [[dh]]
            )
        }
        
        // cube layer
        func makeActivationlayer() -> CoreML_Specification_NeuralNetworkLayer {
            var cubeLayer = CoreML_Specification_UnaryFunctionLayerParams()
            cubeLayer.alpha = 3
            cubeLayer.type = .power
            return nnLayer(
                name: LayerName.activationHidden1,
                inputNames: [LayerName.preActivationReshapeHidden1],
                outputNames: [LayerName.activationHidden1],
                layer: .unary(cubeLayer),
                inputDimensions: [[dh]],
                outputDimensions: [[dh]]
            )
        }
        
        return [
            makeSplitLayer(),
            makeNwDenseLayer(),
            makeNtDenseLayer(),
            makeNlDenseLayer(),
            makeAddLayer(),
            makeBiasLayer(),
            makePreActivationReshapeLayer(),
            makeActivationlayer(),
        ]
    }

    func makeOutputLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        let innerProductLayer = makeInnerProductLayer(numInput: dh, numOutput: numLabels, weights: model.outputLayer.weight.transposed(), bias: model.outputLayer.bias.transposed())
        
        return [nnLayer(
            name: LayerName.output,
            inputNames: [LayerName.activationHidden1],
            outputNames: [LayerName.output],
            layer: .innerProduct(innerProductLayer),
            inputDimensions: [[dh]],
            outputDimensions: [[numLabels]]
        )]
    }
}

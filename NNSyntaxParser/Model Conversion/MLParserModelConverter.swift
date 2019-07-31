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
    let model: ParserModel
    
    func convertToMLModel() throws -> Data {
        var modelBuilder = CoreML_Specification_Model()
        
        modelBuilder.specificationVersion = 4
        modelBuilder.description_p = makeDescription()
        modelBuilder.isUpdatable = false
        modelBuilder.neuralNetworkClassifier = makeNeuralNetworkClassifier()
        return try modelBuilder.serializedData()
    }
    
    // MARK: - NeuralNetworkClassifier
    func makeNeuralNetworkClassifier() -> CoreML_Specification_NeuralNetworkClassifier {
        var nnBuilder = CoreML_Specification_NeuralNetworkClassifier()
        // layers
        var layers = [[CoreML_Specification_NeuralNetworkLayer]]()
        layers.append(makeExpandLayer())
        layers.append(makeEmbeddingLayer())
        layers.append(makeFlattenLayer())
        layers.append(makeCustomHiddenLayer())
        layers.append(makeOutputLayer())
        nnBuilder.layers = layers.flatMap({ $0 })

        nnBuilder.arrayInputShapeMapping = .exactArrayMapping

        // class labels vector
        var classVector = CoreML_Specification_Int64Vector()
        classVector.vector = [Int64](Int64(0) ..< Int64(numLabels))
        nnBuilder.int64ClassLabels = classVector

        nnBuilder.labelProbabilityLayerName = nnBuilder.layers.last!.name
        return nnBuilder
    }
    
    // MARK: - ModelDescription
    private func makeDescription() -> CoreML_Specification_ModelDescription {
        var builder = CoreML_Specification_ModelDescription()
        builder.input = makeInputFeaturesDescription()
        builder.output = makeOutputFeatureDescription()
        builder.metadata = makeMetadata()
        builder.predictedFeatureName = "classLabel"
        builder.predictedProbabilitiesName = "labelProbabilities"
        return builder
    }
    
    private func makeInputFeaturesDescription() -> [CoreML_Specification_FeatureDescription] {
        typealias InputFeature = (name: String, description: String)
        let inputFeatures: [InputFeature] = [
            ("parseState", "features of the current automata state")
        ]
        
        return inputFeatures.map({
            var builder = CoreML_Specification_FeatureDescription()
            builder.name = $0.name
            builder.shortDescription = $0.description
            var arrayType = CoreML_Specification_ArrayFeatureType()
            arrayType.dataType = .int32
            arrayType.shape = [Int64(n_Features)]
            var featureType = CoreML_Specification_FeatureType()
            featureType.type = .multiArrayType(arrayType)
            builder.type = featureType
            return builder
        })
    }
    
    private func makeOutputFeatureDescription() -> [CoreML_Specification_FeatureDescription] {
        typealias OutputFeature = (name: String, description: String, type: CoreML_Specification_FeatureType.OneOf_Type)
        let outputFeatures: [OutputFeature] = [
            ("classLabel", "the next transition that the parser should apply to the current state", .int64Type(CoreML_Specification_Int64FeatureType())),
            ("labelProbabilities",
             "the probabilities of every possible transition",
             .dictionaryType({
                var dict = CoreML_Specification_DictionaryFeatureType()
                dict.int64KeyType = CoreML_Specification_Int64FeatureType()
                return dict}())
            )
        ]

        return outputFeatures.map {
            var builder = CoreML_Specification_FeatureDescription()
            builder.name = $0.name
            builder.shortDescription = $0.description
            var featureType = CoreML_Specification_FeatureType()
            featureType.type = $0.type
            builder.type = featureType
            return builder
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
            name: "parseState_expanded",
            inputNames: ["parseState"],
            outputNames: ["parseState_expanded"],
            layer: .expandDims(expandLayer),
            inputDimensions: [[n_Features]],
            outputDimensions: [[n_Features, 1, 1]]
        )]
    }
    
    func makeEmbeddingLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        var embeddingLayer = CoreML_Specification_EmbeddingNDLayerParams()
        embeddingLayer.vocabSize = UInt64(totalVocabularySize)
        embeddingLayer.embeddingSize = UInt64(embeddingDimension)
        var weights = CoreML_Specification_WeightParams()
        weights.floatValue = model.embeddingLayer.embeddings.transposed().flattened().scalars
        embeddingLayer.weights = weights
        
        return [nnLayer(
            name: "embedding",
            inputNames: ["parseState_expanded"],
            outputNames: ["embedding"],
            layer: .embeddingNd(embeddingLayer),
            inputDimensions: [[n_Features, 1, 1]],
            outputDimensions: [[n_Features, 1, embeddingDimension]]
        )]
    }
    
    func makeFlattenLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        var reshape = CoreML_Specification_ReshapeStaticLayerParams()
        reshape.targetShape = [1, 1, Int64(n_Features * embeddingDimension)]
        
        return [nnLayer(
            name: "flatten_reshape",
            inputNames: ["embedding"],
            outputNames: ["flatten_reshape"],
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
                UInt64(w_WeightDimension[0]),
                UInt64(t_WeightDimension[0]),
                UInt64(l_WeightDimension[0])
            ]
            splitLayer.numSplits = UInt64(splitLayer.splitSizes.count)
            let output = [
                [1, 1, n_wFeatures * embeddingDimension],
                [1, 1, n_tFeatures * embeddingDimension],
                [1, 1, n_lFeatures * embeddingDimension]
            ]
            return nnLayer(
                name: "feature_split",
                inputNames: ["flatten_reshape"],
                outputNames: ["n_wFeature_embeddings", "n_tFeature_embeddings", "n_lFeature_embeddings"],
                layer: .splitNd(splitLayer),
                inputDimensions: [[1, 1, n_Features * embeddingDimension]],
                outputDimensions: output
            )
        }
        
        // n_w Inner Product layer
        var nWInnerProductLayer = CoreML_Specification_InnerProductLayerParams()
        nWInnerProductLayer.inputChannels = UInt64(n_wFeatures * embeddingDimension)
        nWInnerProductLayer.outputChannels = UInt64(dh)
        var w_weights = CoreML_Specification_WeightParams()
        w_weights.floatValue = model.parseLayer.wordWeights.transposed().flattened().scalars
        nWInnerProductLayer.weights = w_weights
        let nnNWInnerProductLayer = nnLayer(
            name: "n_wHiddenLayer_1",
            inputNames: ["n_wFeature_embeddings"],
            outputNames: ["n_wHiddenLayer_1"],
            layer: .innerProduct(nWInnerProductLayer),
            inputDimensions: [[1, 1, n_wFeatures * embeddingDimension]],
            outputDimensions: [[1, 1, dh]]
        )
        
        // n_t InnerProduct layer
        var nTInnerProductLayer = CoreML_Specification_InnerProductLayerParams()
        nTInnerProductLayer.inputChannels = UInt64(n_tFeatures * embeddingDimension)
        nTInnerProductLayer.outputChannels = UInt64(dh)
        var t_weights = CoreML_Specification_WeightParams()
        t_weights.floatValue = model.parseLayer.tagWeights.transposed().flattened().scalars
        nTInnerProductLayer.weights = t_weights
        let nnNTInnerProductLayer = nnLayer(
            name: "n_tHiddenLayer_1",
            inputNames: ["n_tFeature_embeddings"],
            outputNames: ["n_tHiddenLayer_1"],
            layer: .innerProduct(nTInnerProductLayer),
            inputDimensions: [[1, 1, n_tFeatures * embeddingDimension]],
            outputDimensions: [[1, 1, dh]]
        )
        
        // n_l InnerProduct layer
        var nLInnerProductLayer = CoreML_Specification_InnerProductLayerParams()
        nLInnerProductLayer.inputChannels = UInt64(n_lFeatures * embeddingDimension)
        nLInnerProductLayer.outputChannels = UInt64(dh)
        var l_weights = CoreML_Specification_WeightParams()
        l_weights.floatValue = model.parseLayer.labelWeights.transposed().flattened().scalars
        nLInnerProductLayer.weights = l_weights
        let nnNLInnerProductLayer = nnLayer(
            name: "n_lHiddenLayer_1",
            inputNames: ["n_lFeature_embeddings"],
            outputNames: ["n_lHiddenLayer_1"],
            layer: .innerProduct(nLInnerProductLayer),
            inputDimensions: [[1, 1, n_lFeatures * embeddingDimension]],
            outputDimensions: [[1, 1, dh]]
        )
        
        // Add layer
        let nnAddLayer = nnLayer(
            name: "(n_w+n_t+n_l)HiddenLayer_1",
            inputNames: ["n_wHiddenLayer_1", "n_tHiddenLayer_1", "n_lHiddenLayer_1"],
            outputNames: ["(n_w+n_t+n_l)HiddenLayer_1"],
            layer: .add(CoreML_Specification_AddLayerParams()),
            inputDimensions: [[1, 1, dh], [1, 1, dh], [1, 1, dh]],
            outputDimensions: [[1, 1, dh]]
        )
        
        // Bias Layer
        var biasLayer = CoreML_Specification_BiasLayerParams()
        biasLayer.shape = [1, 1, UInt64(dh)]
        var bias_Weights = CoreML_Specification_WeightParams()
        bias_Weights.floatValue = model.parseLayer.bias.transposed().scalars
        biasLayer.bias = bias_Weights
        let nnBiasLayer = nnLayer(
            name: "bias_HiddenLayer_1",
            inputNames: ["(n_w+n_t+n_l)HiddenLayer_1"],
            outputNames: ["bias_HiddenLayer_1"],
            layer: .bias(biasLayer),
            inputDimensions: [[1, 1, dh]],
            outputDimensions: [[1, 1, dh]]
        )
        
        var biasReshape = CoreML_Specification_SqueezeLayerParams()
        biasReshape.squeezeAll = true
        let nnBiasReshape = nnLayer(
            name: "bias_reshape_HiddenLayer_1",
            inputNames: ["bias_HiddenLayer_1"],
            outputNames: ["bias_reshape_HiddenLayer_1"],
            layer: .squeeze(biasReshape),
            inputDimensions: [[1, 1, dh]],
            outputDimensions: [[dh]]
        )
        
        // cube layer
        var cubeLayer = CoreML_Specification_UnaryFunctionLayerParams()
        cubeLayer.alpha = 3
        cubeLayer.type = .power
        let nnCubeActivationLayer = nnLayer(
            name: "cube_activation_HiddenLayer_1",
            inputNames: ["bias_reshape_HiddenLayer_1"],
            outputNames: ["cube_activation_HiddenLayer_1"],
            layer: .unary(cubeLayer),
            inputDimensions: [[dh]],
            outputDimensions: [[dh]]
        )
        
        return [
            makeSplitLayer(),
            nnNWInnerProductLayer,
            nnNTInnerProductLayer,
            nnNLInnerProductLayer,
            nnAddLayer,
            nnBiasLayer,
            nnBiasReshape,
            nnCubeActivationLayer,
        ]
    }

    func makeOutputLayer() -> [CoreML_Specification_NeuralNetworkLayer] {
        var innerProductLayer = CoreML_Specification_InnerProductLayerParams()
        innerProductLayer.inputChannels = UInt64(dh)
        innerProductLayer.outputChannels = UInt64(numLabels)
        var weights = CoreML_Specification_WeightParams()
        weights.floatValue = model.outputLayer.weight.transposed().flattened().scalars
        innerProductLayer.weights = weights
        var bias = CoreML_Specification_WeightParams()
        bias.floatValue = model.outputLayer.bias.transposed().flattened().scalars
        innerProductLayer.bias = bias
        
        return [nnLayer(
            name: "labelProbabilities",
            inputNames: ["cube_activation_HiddenLayer_1"],
            outputNames: ["labelProbabilities"],
            layer: .innerProduct(innerProductLayer),
            inputDimensions: [[dh]],
            outputDimensions: [[numLabels]]
        )]
    }
    
    // MARK: - Layer creation helpers
    private func nnLayer(name: String, inputNames: [String], outputNames: [String], layer: CoreML_Specification_NeuralNetworkLayer.OneOf_Layer, inputDimensions: [[Int]], outputDimensions: [[Int]]) -> CoreML_Specification_NeuralNetworkLayer {
        var nnLayer = CoreML_Specification_NeuralNetworkLayer()
        nnLayer.name = name
        nnLayer.input = inputNames
        nnLayer.output = outputNames
        nnLayer.layer = layer
        nnLayer.inputTensor = inputDimensions.map({ layerDimension(dimension: $0) })
        nnLayer.outputTensor = outputDimensions.map({ layerDimension(dimension: $0) })
        return nnLayer
    }
    
    private func layerDimension(dimension: [Int]) -> CoreML_Specification_Tensor {
        var tensor = CoreML_Specification_Tensor()
        tensor.dimValue = dimension.map({ Int64($0) })
        tensor.rank = UInt32(dimension.count)
        return tensor
    }
}

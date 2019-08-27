//
// MLParserModel.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
class MLParserModelInput : MLFeatureProvider {

    /// features of the current automata state as 39 element vector of 32-bit integers
    var parseState: MLMultiArray

    var featureNames: Set<String> {
        get {
            return ["parseState"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "parseState") {
            return MLFeatureValue(multiArray: parseState)
        }
        return nil
    }
    
    init(parseState: MLMultiArray) {
        self.parseState = parseState
    }
}

/// Model Prediction Output Type
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
class MLParserModelOutput : MLFeatureProvider {

    /// Source provided by CoreML

    private let provider : MLFeatureProvider


    /// the logits distribution of all transition labels as 75 element vector of floats
    lazy var outputTransitionLogits: MLMultiArray = {
        [unowned self] in return self.provider.featureValue(for: "outputTransitionLogits")!.multiArrayValue
    }()!

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(outputTransitionLogits: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["outputTransitionLogits" : MLFeatureValue(multiArray: outputTransitionLogits)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
class MLParserModel {
    var model: MLModel

/// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: MLParserModel.self)
        return bundle.url(forResource: "MLParserModel", withExtension:"mlmodelc")!
    }

    /**
        Construct a model with explicit path to mlmodelc file
        - parameters:
           - url: the file url of the model
           - throws: an NSError object that describes the problem
    */
    init(contentsOf url: URL) throws {
        self.model = try MLModel(contentsOf: url)
    }

    /// Construct a model that automatically loads the model from the app's bundle
    convenience init() {
        try! self.init(contentsOf: type(of:self).urlOfModelInThisBundle)
    }

    /**
        Construct a model with configuration
        - parameters:
           - configuration: the desired model configuration
           - throws: an NSError object that describes the problem
    */
    convenience init(configuration: MLModelConfiguration) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct a model with explicit path to mlmodelc file and configuration
        - parameters:
           - url: the file url of the model
           - configuration: the desired model configuration
           - throws: an NSError object that describes the problem
    */
    init(contentsOf url: URL, configuration: MLModelConfiguration) throws {
        self.model = try MLModel(contentsOf: url, configuration: configuration)
    }

    /**
        Make a prediction using the structured interface
        - parameters:
           - input: the input to the prediction as MLParserModelInput
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as MLParserModelOutput
    */
    func prediction(input: MLParserModelInput) throws -> MLParserModelOutput {
        return try self.prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface
        - parameters:
           - input: the input to the prediction as MLParserModelInput
           - options: prediction options
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as MLParserModelOutput
    */
    func prediction(input: MLParserModelInput, options: MLPredictionOptions) throws -> MLParserModelOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return MLParserModelOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface
        - parameters:
            - parseState: features of the current automata state as 39 element vector of 32-bit integers
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as MLParserModelOutput
    */
    func prediction(parseState: MLMultiArray) throws -> MLParserModelOutput {
        let input_ = MLParserModelInput(parseState: parseState)
        return try self.prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface
        - parameters:
           - inputs: the inputs to the prediction as [MLParserModelInput]
           - options: prediction options
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as [MLParserModelOutput]
    */
    func predictions(inputs: [MLParserModelInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [MLParserModelOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [MLParserModelOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  MLParserModelOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}

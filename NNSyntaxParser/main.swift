//
//  main.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-09.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow
import CoreML
import NaturalLanguage
import Python

// Check if data requirements are present
print("Checking data requirements...")
do {
    try DataRequirementsChecker().checkRequirements()
} catch DataRequirementsCheckerError.requirementsNotMet(let unmetRequirements) {
    print("Downloading unmet requirements")
    let downloader = Downloader()
    unmetRequirements.enumerated().forEach {
        print("Downloading requirement \($0.offset + 1)/\(unmetRequirements.count): \($0.element.file.name)")
        downloader.download(requirement: $0.element.file)
    }
}
print("All data requirements have been met")

print("Extracting files from downloaded data")
print("Extracting files from downloaded UD archive...")
UDEnglishExtractor().extract(from: DataRequirement.udTreebank.file.name)
print("Extracting files from downloaded Glove archive...")
GloveExtractor().extract(from: DataRequirement.glove.file.name)
print("Extraction completed")

print("populating token embeddings...")
let vocabularyEmbeddings = EmbeddingsReader.dependencyEmbedding(from: GloveExtractor.extractedResourcePath)
print("token embeddings populated")

let featureProvider = vocabularyEmbeddings.featureProvider
let serializer = ModelSerializer()

var savedModel: TFParserModel?
var savedEpoch: Int?
for i in (1...3).reversed() {
    let modelName = TFParseTrainer.savedModelName(epoch: i)
    if serializer.modelExists(name: modelName) {
        savedModel = try! serializer.loadModel(name: modelName)
        savedEpoch = i
        break
    }
}

// MARK: Model training
let trainer = TFParseTrainer(serializer: serializer,
                           explorationEpochThreshold: 1,
                           explorationProbability: 0.9,
                           featureProvider: featureProvider,
                           model: savedModel ?? TFParserModel(embeddings: vocabularyEmbeddings.embedding))
let startDate = Date()
print("beginning training...")
print("loading train examples...")
let trainExamples = UDReader.readTrainData()
print("training model with \(trainExamples.count) examples...")
let history = trainer.train(trainSet: trainExamples, batchSize: 32, startEpoch: (savedEpoch ?? 0) + 1, epochs: 50)
print("training done. Took \(Date().timeIntervalSince(startDate)/3600) hours")

// MARK: - Plot training performance
let plt = Python.import("matplotlib.pyplot")
plt.figure(figsize: [12, 8])

let accuracyAxes = plt.subplot(2, 1, 1)
accuracyAxes.set_ylabel("Accuracy")
accuracyAxes.plot(history.trainAccuracies)

let lossAxes = plt.subplot(2, 1, 2)
lossAxes.set_ylabel("Loss")
lossAxes.set_xlabel("Epoch")
lossAxes.plot(history.trainLoss)

plt.show()

// MARK: - Model testing
extension TFParserModel: ParserModel {
    func transitionProbabilities(for features: [Int32]) throws -> [Int : Float] {
        let embeddingsInput = EmbeddingInput(indices: Tensor<Int32>(ShapedArray<Int32>(shape: [1, features.count], scalars: features)))
        let prediction = self(embeddingsInput)
        return prediction[0].scalars.enumerated().reduce([Int: Float]()) {
            var result = $0
            result[$1.offset] = $1.element
            return result
        }
    }
}

func test(parser: Parser, examples: [ParseExample]) -> Float {
    var results = [Float]()
    for example in examples {
        let answer = try! parser.parse(sentence: example.sentence)
        let accuracy = Float(answer.heads.enumerated().filter({ $0.element?.head == example.goldArcs[$0.offset]?.head && $0.element?.relationship == example.goldArcs[$0.offset]?.relationship }).count) / Float(answer.heads.count)
        results.append(accuracy)
    }

    return results.reduce(0, +) / Float(results.count)
}

print("testing model...")
print("loading test examples...")
let testExamples = UDReader.readTestData().shuffled()
print("testing model with \(testExamples.count) examples...")
let accuracy = test(
    parser: Parser(model: savedModel!, featureProvider: featureProvider),
    examples: testExamples
)
print("testing done. Accuracy: \(accuracy * 100.0)%")

let shouldSaveFinalModel = true
if shouldSaveFinalModel {
    print("Saving final trained model")
    try! serializer.save(model: trainer.model, to: "FINAL_trained_model")
    print("Model saved...")
    print("Done.")
}

// MARK: - model conversion
let shouldConvert = true
if shouldConvert {
    let converter = MLParserModelConverter(model: savedModel!)
    let converted = try! converter.convertToMLModel()
    let downloadsURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!.appendingPathComponent("MLParserModel").appendingPathExtension("mlmodel")
    try! converted.write(to: downloadsURL, options: .atomic)
}

// MARK: - test converted model
//extension MLParserModel: ParserModel {
//    func transitionProbabilities(for features: [Int32]) throws -> [Int : Float] {
//        let arrayInput = try MLMultiArray(features)
//        let prediction = try self.prediction(input: MLParserModelInput(parseState: arrayInput))
//        return (0..<numLabels).reduce([Int:Float]()) {
//            var result = $0
//            result[$1] = prediction.outputTransitionLogits[[NSNumber(value: $1)]].floatValue
//            return result
//        }
//    }
//}
//
//let convertedAccuracy = test(
//    parser: Parser(model: MLParserModel(), featureProvider: featureProvider),
//    examples: testExamples
//)
//print("testing converted done. Accuracy: \(convertedAccuracy * 100.0)%")

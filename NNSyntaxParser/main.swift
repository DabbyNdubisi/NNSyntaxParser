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

let batchSize = 4
let epochs = 20

let featureProvider = vocabularyEmbeddings.featureProvider
let serializer = ModelSerializer(location: FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!.appendingPathComponent("trained-models-batch-\(batchSize)"))

// MARK: Model training
let trainer = TFParseTrainer(serializer: serializer,
                           explorationEpochThreshold: 10,
                           explorationProbability: 1.0,
                           regularizerParameter: 0.00000001,
                           featureProvider: featureProvider,
                           model: TFParserModel(embeddings: vocabularyEmbeddings.embedding))
let startDate = Date()
print("beginning training...")
print("loading train examples...")
let trainExamples = UDReader.readTrainData()
print("loading validtion examples...")
let validationExamples = UDReader.readValidationData()
print("training model with \(trainExamples.count) examples...")
let (train, validation) = trainer.train(trainSet: trainExamples, validationSet: validationExamples, batchSize: batchSize, epochs: epochs, retrieveCheckpoint: true, saveCheckpoints: true)
print("training done. Took \(Date().timeIntervalSince(startDate)/3600) hours")

 MARK: - Plot training performance
let plt = Python.import("matplotlib.pyplot")
plt.figure(figsize: [12, 8])

let accuracyAxes = plt.subplot(2, 1, 1)
accuracyAxes.set_ylabel("Accuracy")
accuracyAxes.plot(train.accuracyResults)
accuracyAxes.plot(validation.accuracyResults)
accuracyAxes.legend(["train", "validation"], loc:"upper right")

let lossAxes = plt.subplot(2, 1, 2)
lossAxes.set_ylabel("Loss")
lossAxes.set_xlabel("Epoch")
lossAxes.plot(train.lossResults)
lossAxes.plot(validation.lossResults)
lossAxes.legend(["train", "validation"], loc:"upper right")

plt.show()

// MARK: - Model testing
print("retrieving best model")
let bestModel = try! serializer.loadModel(name: TFParseTrainer.Constants.bestTrainedModelName)
print("testing model...")
print("loading test examples...")
let testExamples = UDReader.readTestData().shuffled()
print("testing model with \(testExamples.count) examples...")
let accuracy = test(
    parser: Parser(model: bestModel, featureProvider: featureProvider),
    examples: testExamples
)
print("testing done. Accuracy: \(accuracy * 100.0)%")

//// MARK: - model conversion
//let shouldConvert = true
//if shouldConvert {
//    let converter = MLParserModelConverter(model: bestModel!)
//    let converted = try! converter.convertToMLModel()
//    let downloadsURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!.appendingPathComponent("MLParserModel").appendingPathExtension("mlmodel")
//    try! converted.write(to: downloadsURL, options: .atomic)
//}

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

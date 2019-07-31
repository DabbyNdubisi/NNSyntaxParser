//
//  main.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-09.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow

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
print("Extracting files from downloaded PTB archive...")
PTBDependencyExtractor().extract(from: DataRequirement.depPennTreebank.file.name)
print("Extraction completed")

print("populating token embeddings...")
let vocabularyEmbeddings = EmbeddingsReader.dependencyEmbedding(from: GloveExtractor.extractedResourcePath)
print("token embeddings populated")

let featureProvider = vocabularyEmbeddings.featureProvider
let serializer = ModelSerializer()

var savedModel: ParserModel?
var savedEpoch: Int?
for i in (1...3).reversed() {
    let modelName = ParseTrainer.savedModelName(epoch: i)
    if serializer.modelExists(name: modelName) {
        savedModel = try! serializer.loadModel(name: modelName)
        savedEpoch = i
        break
    }
}

let trainer = ParseTrainer(serializer: serializer,
                           explorationEpochThreshold: 1,
                           explorationProbability: 0.9,
                           featureProvider: featureProvider,
                           model: savedModel ?? ParserModel(embeddings: vocabularyEmbeddings.embedding))

//let startDate = Date()
//print("beginning training...")
//print("loading train examples...")
//let trainExamples = UDReader.readTrainData()
//print("training model with \(trainExamples.count) examples...")
//trainer.fasterTrain(examples: trainExamples, batchSize: 100, startEpoch: (savedEpoch ?? 0) + 1, epochs: 3)
//print("training done. Took \(Date().timeIntervalSince(startDate)/3600) hours")
//
print("testing model...")
print("loading test examples...")
let testExamples = UDReader.readTestData().shuffled()
print("testing model with \(testExamples.count) examples...")
let accuracy = trainer.test(examples: testExamples)
print("testing done. Accuracy: \(accuracy * 100.0)%")
//
//print("Saving final trained model")
//try! serializer.save(model: trainer.model, to: "FINAL_trained_model")
//print("Model saved...")
//
//print("Done.")


// MARK: - model conversion
import CoreML
import NaturalLanguage

//let converter = MLParserModelConverter(model: savedModel!)
//let converted = try! converter.convertToMLModel()
//let downloadsURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!.appendingPathComponent("MLParserModel").appendingPathExtension("mlmodel")
//try! converted.write(to: downloadsURL, options: .atomic)

//extension TransitionFeatureProvider {
//    func mlfeatures(state: ParserAutomata, sentence: String) -> MLParserModelInput {
//        let feats = self.features(for: state, sentence: sentence)
//        let array = try! MLMultiArray(feats)
//        return MLParserModelInput(parseState: array)
//    }
//}
//
//class Parser {
//    private let rootPrefix = "<ROOT>"
//    private let model: MLParserModel
//    private let featureProvider: TransitionFeatureProvider
//    private let tagger: NLTagger = NLTagger(tagSchemes: [.lexicalClass])
//
//    init(model: MLParserModel, featureProvider: TransitionFeatureProvider) {
//        self.featureProvider = featureProvider
//        self.model = model
//    }
//
//    func parse(sentence: String) -> Parse {
//        guard let rootPrefixRange = sentence.range(of: rootPrefix) else {
//            return parse(sentence: "\(rootPrefix) \(sentence)")
//        }
//
//        let range = sentence.range(of: sentence.suffix(from: rootPrefixRange.upperBound))!
//        tagger.string = sentence
//        let rootToken = Token(i: 0, sentenceRange: rootPrefixRange, posTag: .root)
//        let buffer =
//            tagger.tags(in: range, unit: .word, scheme: .lexicalClass, options: [.omitWhitespace]).enumerated().map({ Token(i: $0.offset + 1, sentenceRange: $0.element.1, posTag: POSTag(nlTag: $0.element.0!)! ) })
//        var state = ParserAutomata(rootToken: rootToken, buffer: buffer)
//
//        while !state.isTerminal {
//            let valids = state.validTransitions()
//            let prediction = try! model.prediction(input: featureProvider.mlfeatures(state: state, sentence: sentence))
//            let bestPrediction = valids.max(by: {
//                prediction.labelProbabilities[Int64($0.rawValue)]! < prediction.labelProbabilities[Int64($1.rawValue)]!
//            })!
//            state.apply(transition: bestPrediction)
//        }
//
//        return state.parse
//    }
//}
//
//let parser = Parser(model: try! MLParserModel(), featureProvider: featureProvider)
//
//func test(examples: [ParseExample]) -> Float {
//    var results = [Float]()
//    for example in examples {
//        let answer = parser.parse(sentence: example.sentence)
//        let accuracy = Float(answer.heads.enumerated().filter({ $0.element?.head == example.goldArcs[$0.offset]?.head && $0.element?.relationship == example.goldArcs[$0.offset]?.relationship }).count) / Float(answer.heads.count)
//        results.append(accuracy)
//    }
//
//    return results.reduce(0, +) / Float(results.count)
//}
//let convertedAccuracy = test(examples: testExamples)
//print("testing converted done. Accuracy: \(convertedAccuracy * 100.0)%")
// 21.9% (TF model test)
// 23.946053% (converted)
//testing done. Accuracy: 52.958836%
//2019-07-28 14:37:15.624664-0400 NNSyntaxParser[31205:939577] Metal API Validation Enabled
//2019-07-28 14:37:15.656151-0400 NNSyntaxParser[31205:946592] flock failed to lock maps file: errno = 35
//2019-07-28 14:37:15.656823-0400 NNSyntaxParser[31205:946592] flock failed to lock maps file: errno = 35
//testing converted done. Accuracy: 21.453232%

// MARK: - testing layers
//let testExamples = UDReader.readTestData().shuffled()
extension TransitionFeatureProvider {
    func mlfeatures(state: ParserAutomata, sentence: String) -> MLParserModelInput {
        let feats = self.features(for: state, sentence: sentence)
        let array = try! MLMultiArray(feats)
        return MLParserModelInput(parseState: array)
    }
}

class Parser {
    private let rootPrefix = "<ROOT>"
    private let model: MLParserModel
    private let featureProvider: TransitionFeatureProvider
    private let tagger: NLTagger = NLTagger(tagSchemes: [.lexicalClass])

    init(model: MLParserModel, featureProvider: TransitionFeatureProvider) {
        self.featureProvider = featureProvider
        self.model = model
    }

    func parse(sentence: String) -> Parse {
        guard let rootPrefixRange = sentence.range(of: rootPrefix) else {
            return parse(sentence: "\(rootPrefix) \(sentence)")
        }

        let range = sentence.range(of: sentence.suffix(from: rootPrefixRange.upperBound))!
        tagger.string = sentence
        let rootToken = Token(i: 0, sentenceRange: rootPrefixRange, posTag: .root)
        let buffer =
            tagger.tags(in: range, unit: .word, scheme: .lexicalClass, options: [.omitWhitespace]).enumerated().map({ Token(i: $0.offset + 1, sentenceRange: $0.element.1, posTag: POSTag(nlTag: $0.element.0!)! ) })
        var state = ParserAutomata(rootToken: rootToken, buffer: buffer)

        while !state.isTerminal {
            let valids = state.validTransitions()
            let prediction = try! model.prediction(input: featureProvider.mlfeatures(state: state, sentence: sentence))
            let bestPrediction = valids.max(by: {
                prediction.labelProbabilities[Int64($0.rawValue)]! < prediction.labelProbabilities[Int64($1.rawValue)]!
            })!
            state.apply(transition: bestPrediction)
        }
        return state.parse
    }
}

let parser = Parser(model: MLParserModel(), featureProvider: featureProvider)
func test(examples: [ParseExample]) -> Float {
    var results = [Float]()
    for example in examples {
        let answer = parser.parse(sentence: example.sentence)
        let accuracy = Float(answer.heads.enumerated().filter({ $0.element?.head == example.goldArcs[$0.offset]?.head && $0.element?.relationship == example.goldArcs[$0.offset]?.relationship }).count) / Float(answer.heads.count)
        results.append(accuracy)
    }

    return results.reduce(0, +) / Float(results.count)
}
//parser.parse(sentence: testExamples[0].sentence)
let convertedAccuracy = test(examples: testExamples)
print("testing converted done. Accuracy: \(convertedAccuracy * 100.0)%")

//
//  TFParseTrainer.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-14.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow
import NaturalLanguage

class TFParseTrainer {
    enum Constants {
        static let trainModelCheckpointName = "trainedModel"
        static let bestTrainedModelName = "trainedModel-best"
    }
    
    struct History {
        var accuracyResults: [Float] = []
        var lossResults: [Float] = []
    }
    
    static func savedModelName(epoch: Int) -> String {
        return "\(Constants.trainModelCheckpointName)_\(epoch)epoch"
    }
    
    var model: TFParserModel
    let serializer: ModelSerializer
    let featureProvider: TransitionFeatureProvider
    let explorationEpochThreshold: Int
    let explorationProbability: Float
    let regularizerParameter: Float
    
    private let saveQueue = DispatchQueue(label: "save.queue")
    
    init(serializer: ModelSerializer, explorationEpochThreshold: Int, explorationProbability: Float, regularizerParameter: Float, featureProvider: TransitionFeatureProvider, model: TFParserModel) {
        self.serializer = serializer
        self.explorationEpochThreshold = explorationEpochThreshold
        self.explorationProbability = explorationProbability
        self.regularizerParameter = regularizerParameter
        self.featureProvider = featureProvider
        self.model = model
    }
    
    func train(trainSet: [ParseExample], validationSet: [ParseExample]? = nil, batchSize: Int = 32, startEpoch: Int = 1, epochs: Int, retrieveCheckpoint: Bool, saveCheckpoints: Bool) -> (trainHistory: History, validationHistory: History) {
        // retreive checkpoints
        if retrieveCheckpoint, let modelToRetrieve = retrieveLastSavedModel(totalEpochs: epochs) {
            model = modelToRetrieve.model
            return train(trainSet: trainSet, validationSet: validationSet, batchSize: batchSize, startEpoch: modelToRetrieve.epoch + 1, epochs: epochs, retrieveCheckpoint: false, saveCheckpoints: saveCheckpoints)
        }
        
        var trainHistory = History()
        var validationHistory = History()
        guard startEpoch <= epochs else {
            return (trainHistory, validationHistory)
        }

        let optimizer = Adam(for: model, learningRate: 0.001)
        var bestLas: Float = 0.0
        for epoch in startEpoch...epochs {
            let epochStartDate = Date()
            
            // train
            var (epochAccuracy, epochLoss, batchCount) = evaluate(on: trainSet, optimizer: optimizer, batchSize: batchSize, explorer: { guess, correct in self.explore(currentEpoch: epoch, guess: guess, correct: correct) })
            // track training metrics
            epochAccuracy /= Float(batchCount)
            epochLoss /= Float(batchCount)
            trainHistory.accuracyResults.append(epochAccuracy)
            trainHistory.lossResults.append(epochLoss)
            print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy), Duration: \(Date().timeIntervalSince(epochStartDate)/3600) hours")
            
            // validate on validation set
            if let validationSet = validationSet {
                var (validationAccuracy, validationLoss, validationBatchCount) = evaluate(on: validationSet, batchSize: validationSet.count, explorer: { _, correct in correct })
                validationAccuracy /= Float(validationBatchCount)
                validationLoss /= Float(validationBatchCount)
                validationHistory.accuracyResults.append(validationAccuracy)
                validationHistory.lossResults.append(validationLoss)
                print("Epoch \(epoch): Validation Loss: \(validationLoss), Validation Accuracy: \(validationAccuracy)")
                
                let parser = Parser(model: model, featureProvider: featureProvider)
                let devLas = test(parser: parser, examples: validationSet)
                print("Epoch \(epoch) Validation LAS: \(devLas)")
                
                if devLas > bestLas {
                    bestLas = devLas
                    
                    // save the new best model at this checkpoint
                    guard saveCheckpoints else { continue }
                    print("Saving model with best LAS (\(epoch) epoch)")
                    save(model: model, name: TFParseTrainer.Constants.bestTrainedModelName) {
                        print("best model saved")
                    }
                }
            }
            
            // save checkpoint for model
            guard saveCheckpoints else { continue }
            print("Saving model checkpoint after \(epoch) epoch")
            save(model: model, name: TFParseTrainer.savedModelName(epoch: epoch)) {
                print("trained model saved")
            }
        }
                
        return (trainHistory, validationHistory)
    }
    
    private func evaluate(on dataSet: [ParseExample], optimizer: Adam<TFParserModel>? = nil, batchSize: Int, explorer: @escaping (Transition, Transition) -> Transition) -> (accuracy: Float, loss: Float, batchCount: Int) {
        // metrics
        var epochLoss: Float = 0
        var epochAccuracy: Float = 0
        var ignoredSentencesCount: Int = 0
        var batchCount: Int = 0
        
        let shuffled = dataSet.shuffled()
        var shuffledIterator = shuffled.makeIterator()
        var workingExamples = (0..<batchSize).map({ _ in shuffledIterator.next() })
        var states = (0..<workingExamples.count).map({
            (workingExamples[$0] == nil) ? nil : ParserAutomata(rootPrefix: Parser.rootPrefix, sentence: workingExamples[$0]!.sentence)
        })
        
        var nonTerminalStatesIndices = self.nonTerminalStateIndices(for: states)
        while !nonTerminalStatesIndices.isEmpty {
            var ignoredStateIndices = Set<Int>()
            let stateWithCorrects: [(stateIdx: Int, corrects: [Transition])] = {
                var statesWithCorrects = [(stateIdx: Int, corrects: [Transition])]()
                for stateIdx in nonTerminalStatesIndices {
                    let corrects = states[stateIdx]!.correctTransition(goldArcs: workingExamples[stateIdx]!.goldArcs)
                    if corrects.isEmpty {
                        ignoredStateIndices.insert(stateIdx)
                    } else {
                        statesWithCorrects.append((stateIdx, corrects))
                    }
                }
                return statesWithCorrects
            }()
            
            if !ignoredStateIndices.isEmpty {
                // If there are no correct transitions, it means that the gold tree is not reachable
                // via our transition system. In such scenarios, it doesn't make sense to continue
                // training on the current example as it could confuse the model more.
                ignoredSentencesCount += ignoredStateIndices.count
                print("Ignoring \(ignoredStateIndices.count) sentence as gold tree cannot be attained")
                print("*** \(ignoredSentencesCount) SENTENCES IGNORED SO FAR")
            }
            
            if stateWithCorrects.count > 0 {
                var transitionBatch: TransitionBatch = {
                    let feats = stateWithCorrects.map({ featureProvider.features(for: states[$0.stateIdx]!) })
                    let embeddingInput = EmbeddingInput(indices: Tensor<Int32>(ShapedArray<Int32>(shape: [stateWithCorrects.count, feats[0].count], scalars: feats.flatMap({ $0 }))))
                    return TransitionBatch(
                        features: embeddingInput,
                        transitionLabels: Tensor<Int32>([0]) // will be set later
                    )
                }()
                
                let valids = stateWithCorrects.map({
                    states[$0.stateIdx]!.validTransitions()
                })
                let guesses = model(transitionBatch.features) // predictions
                let bestGuesses = (0..<valids.count).map({ indexInBatch in
                    valids[indexInBatch].max(by: { guesses[indexInBatch][$0.rawValue] < guesses[indexInBatch][$1.rawValue] })!
                })
                transitionBatch.transitionLabels = Tensor<Int32>((0..<stateWithCorrects.count).map({ indexInBatch in
                    return Int32(stateWithCorrects[indexInBatch].corrects.max(by: { guesses[indexInBatch][$0.rawValue] < guesses[indexInBatch][$1.rawValue] })!.rawValue)
                }))
                
                let (loss, gradient) = lossWithGradient(batch: transitionBatch)
                if let optimizer = optimizer {
                    optimizer.update(&model.allDifferentiableVariables, along: gradient)
                }
                let logits = model(transitionBatch.features)
                epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: transitionBatch.transitionLabels)
                epochLoss += loss.scalarized()
                batchCount += 1
                
                // apply transition to state
                for i in 0 ..< stateWithCorrects.count {
                    let (stateIdx, _) = stateWithCorrects[i]
                    states[stateIdx]!.apply(transition: explorer(bestGuesses[i], Transition(rawValue: Int(transitionBatch.transitionLabels[i].scalar!))!))
                }
            }
            
            // update working to remove ignoredIndices and new terminal states batch
            for idx in 0..<workingExamples.count {
                let shouldUseNewExample = ignoredStateIndices.contains(idx) || (states[idx]?.isTerminal ?? true)
                if shouldUseNewExample {
                    let newExample = shuffledIterator.next()
                    workingExamples[idx] = newExample
                    states[idx] = newExample != nil ? ParserAutomata(rootPrefix: Parser.rootPrefix, sentence: newExample!.sentence) : nil
                }
            }
        
            // Set new nonTerminalStateIndices
            nonTerminalStatesIndices = self.nonTerminalStateIndices(for: states)
        }
        
        return (epochAccuracy, epochLoss, batchCount)
    }
    
    private func save(model: TFParserModel, name: String, completion: @escaping () -> Void) {
        saveQueue.async {
            try! self.serializer.save(model: model, to: name)
            completion()
        }
    }
    
    private func retrieveLastSavedModel(totalEpochs: Int) -> (model: TFParserModel, epoch: Int)? {
        var savedModel: TFParserModel?
        var savedEpoch: Int?
        for i in (1...totalEpochs).reversed() {
            let modelName = TFParseTrainer.savedModelName(epoch: i)
            if serializer.modelExists(name: modelName) {
                savedModel = try! serializer.loadModel(name: modelName)
                savedEpoch = i
                break
            }
        }
        
        if let savedModel = savedModel, let savedEpoch = savedEpoch {
            return (savedModel, savedEpoch)
        } else {
            return nil
        }
    }
    
    private func nonTerminalStateIndices(for states: [ParserAutomata?]) -> [Int] {
        return (0..<states.count).compactMap({
            (states[$0]?.isTerminal ?? true) ? nil : $0
        })
    }
    
    private func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
        return Tensor<Float>(predictions .== truths).mean().scalarized()
    }
    
    private func lossWithGradient(batch: TransitionBatch) -> (loss: Tensor<Float>, gradient: TFParserModel.TangentVector) {
        let (loss, grad) = model.valueWithGradient { (model: TFParserModel) -> Tensor<Float> in
            let logits = model(batch.features)
            let l2Loss = withoutDerivative(at: model.l2Loss)
            let crossEntropyLoss = softmaxCrossEntropy(logits: logits, labels: batch.transitionLabels)
            return crossEntropyLoss + (l2Loss * self.regularizerParameter)
        }
        
        return (loss, grad)
    }
    
    private func explore(currentEpoch: Int, guess: Transition, correct: Transition) -> Transition {
        if currentEpoch > explorationEpochThreshold && Float.random(in: 0...1) < explorationProbability {
            return guess
        } else {
            return correct
        }
    }
}

// MARK: - Dynamic oracle for training
extension ParserAutomata {
    /// implementation of a dynamic oracle
    func correctTransition(goldArcs: [Dependency?]) -> [Transition] {
        var shiftIsValid = false
        var leftIsValid = false
        var rightIsValid = false
        let valids = Set(validTransitions())
        if valids.first(where: { $0.isShift }) != nil { shiftIsValid = true }
        if valids.first(where: { $0.isLeft }) != nil { leftIsValid = true }
        if valids.first(where: { $0.isRight }) != nil { rightIsValid = true }
        
        // Return zero cost moves that could lead to the best optimal tree reachable from the current state
        var zeroCostTransitions = [Transition]()
        if shiftIsValid && isShiftZeroCost(goldArcs: goldArcs) {
            zeroCostTransitions.append(.shift)
        }
        if rightIsValid && isRightZeroCost(goldArcs: goldArcs) {
            let transitions: [Transition] = (goldArcs[stack.last!.i]!.head == stack[stack.count - 2].i) ?
                [.right(relation: goldArcs[stack.last!.i]!.relationship)] :
                [Transition](valids.filter { $0.isRight })
            
            zeroCostTransitions += transitions
        }
        if leftIsValid && isLeftZeroCost(goldArcs: goldArcs) {
            let transitions: [Transition] = (goldArcs[stack.last!.i]!.head == buffer.last!.i) ?
                [.left(relation: goldArcs[stack.last!.i]!.relationship)] :
                [Transition](valids.filter { $0.isLeft })
            
            zeroCostTransitions += transitions
        }
        return zeroCostTransitions
    }
    
    /**
     Adding the arc (s1, s0) and popping s0 from the stack means that s0 will not be able
     to acquire heads or deps from B.  The cost is the number of arcs in gold_conf of the form
     (s0, d) and (h, s0) where h, d in B.  For non-zero cost moves, we are looking simply for
     (s0, b) or (b, s0) for all b in B
     */
    func isRightZeroCost(goldArcs: [Dependency?]) -> Bool {
        return !dependencyExists(token: stack.last!, others: buffer, in: goldArcs)
    }
    
    /**
     Adding the arc (b, s0) and popping s0 from the stack means that s0 will not be able to acquire
     heads from H = {s1} U B and will not be able to acquire dependents from B U b, therefore the cost is
     the number of arcs in T of form (s0, d) or (h, s0), h in H, d in D
     To have cost, then, only one instance must occur.
     
     Therefore left is zero cost if no depency exists between s0 & (s1 U B)
     */
    func isLeftZeroCost(goldArcs: [Dependency?]) -> Bool {
        if let s1 = stack.count >= 2 ? stack[stack.count - 2] : nil, goldArcs[stack.last!.i]?.head == s1.i {
            return false
        } else if let b = buffer.last, goldArcs[b.i]?.head == stack.last!.i {
            return false
        } else {
            return !dependencyExists(token: stack.last!, others: (buffer.count >= 2 ? [Token](buffer[0..<(buffer.count - 1)]) : []), in: goldArcs)
        }
    }
    
    /**
     Pushing b onto the stack means that b will not be able to acquire
     heads from H = {s1} U S and will not be able to acquire deps from
     D = {s0, s1} U S
     */
    func isShiftZeroCost(goldArcs: [Dependency?]) -> Bool {
        if let s0 = stack.last, goldArcs[s0.i]?.head == buffer.last!.i {
            return false
        } else {
            return !dependencyExists(token: buffer.last!, others: (stack.count >= 2 ? [Token](stack[0..<(stack.count - 1)]) : []), in: goldArcs)
        }
    }
    
    /// Returns `true` if a dependency exists between `token` and any element in `others` or
    /// `false` otherwise
    private func dependencyExists(token: Token, others: [Token], in goldArcs: [Dependency?]) -> Bool {
        for other in others {
            if goldArcs[other.i]?.head == token.i || goldArcs[token.i]?.head == other.i {
                return true
            }
        }
        return false
    }
}


// MARK: - Private helpers
struct TransitionBatch {
    /// [batch, featcount] tensor of features
    let features: EmbeddingInput
    /// [batch] tensor of transitions
    var transitionLabels: Tensor<Int32>
}

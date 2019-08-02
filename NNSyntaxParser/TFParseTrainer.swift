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
    }
    
    static func savedModelName(epoch: Int) -> String {
        return "\(Constants.trainModelCheckpointName)_\(epoch)epoch"
    }
    
    var model: TFParserModel
    let serializer: ModelSerializer
    let featureProvider: TransitionFeatureProvider
    let tagger: NLTagger = NLTagger(tagSchemes: [.lexicalClass])
    let explorationEpochThreshold: Int
    let explorationProbability: Float
    
    init(serializer: ModelSerializer, explorationEpochThreshold: Int, explorationProbability: Float, featureProvider: TransitionFeatureProvider, model: TFParserModel) {
        self.serializer = serializer
        self.explorationEpochThreshold = explorationEpochThreshold
        self.explorationProbability = explorationProbability
        self.featureProvider = featureProvider
        self.model = model
    }
    
    func train(trainSet: [ParseExample], batchSize: Int = 32, startEpoch: Int = 1, epochs: Int) -> (trainAccuracies: [Float], trainLoss: [Float]) {
        guard startEpoch <= epochs else {
            return ([], [])
        }
        
        let queue = DispatchQueue(label: "training_queue")
        let group = DispatchGroup()
        let optimizer = Adam(for: model, learningRate: 0.01)
        
        var trainAccuracyResults: [Float] = []
        var trainLossResults: [Float] = []
        for epoch in startEpoch...epochs {
            var epochLoss: Float = 0
            var epochAccuracy: Float = 0
            var ignoredSentencesCount: Int = 0
            var batchCount: Int = 0
            let epochStartDate = Date()
            
            let shuffled = trainSet.shuffled()
            var shuffledIterator = shuffled.makeIterator()
            var workingExamples = (0..<batchSize).map({ _ in shuffledIterator.next() })
            var states: [ParserAutomata?] = {
                var states = [ParserAutomata?](repeating: nil, count: workingExamples.count)
                for i in 0..<workingExamples.count {
                    group.enter()
                    queue.async { [tagger] in
                        states[i] = (workingExamples[i] == nil) ? nil : ParserAutomata(tagger: tagger, rootPrefix: Parser.rootPrefix, sentence: workingExamples[i]!.sentence)
                        group.leave()
                    }
                }
                group.wait()
                return states
            }()
            
            var nonTerminalStatesIndices = self.nonTerminalStateIndices(for: states)
            repeat {
                var ignoredStateIndices = Set<Int>()
                let stateWithCorrects: [(stateIdx: Int, corrects: [Transition])] = {
                    var statesWithCorrects = [(stateIdx: Int, corrects: [Transition])]()
                    for i in 0 ..< nonTerminalStatesIndices.count {
                        group.enter()
                        queue.async {
                            let correct = states[nonTerminalStatesIndices[i]]!.correctTransition(goldArcs: workingExamples[nonTerminalStatesIndices[i]]!.goldArcs)
                            if correct.isEmpty {
                                ignoredStateIndices.insert(nonTerminalStatesIndices[i])
                            } else {
                                statesWithCorrects.append((nonTerminalStatesIndices[i], correct))
                            }
                            group.leave()
                        }
                    }
                    group.wait()
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
                
                var transitionBatch: TransitionBatch = {
                    var feats = [[Int32]](repeating: [], count: stateWithCorrects.count)
                    for i in 0..<stateWithCorrects.count {
                        group.enter()
                        queue.async {
                            feats[i] = self.featureProvider.features(for: states[stateWithCorrects[i].stateIdx]!, sentence: workingExamples[stateWithCorrects[i].stateIdx]!.sentence)
                            group.leave()
                        }
                    }
                    group.wait()
                    
                    let embeddingInput = EmbeddingInput(indices: Tensor<Int32>(ShapedArray<Int32>(shape: [stateWithCorrects.count, feats[0].count], scalars: feats.flatMap({ $0 }))))
                    return TransitionBatch(
                        features: embeddingInput,
                        transitionLabels: Tensor<Int32>([0]) // will be set later
                    )
                }()
                
                let valids: [[Transition]] = {
                    var valids = [[Transition]](repeating: [], count: stateWithCorrects.count)
                    for i in 0..<stateWithCorrects.count {
                        group.enter()
                        queue.async {
                            valids[i] = states[stateWithCorrects[i].stateIdx]!.validTransitions()
                            group.leave()
                        }
                    }
                    group.wait()
                    return valids
                }()
                
                let guesses = model(transitionBatch.features) // predictions
                
                let bestGuesses = (0..<valids.count).map({ indexInBatch in
                    valids[indexInBatch].max(by: { guesses[indexInBatch][$0.rawValue] < guesses[indexInBatch][$1.rawValue] })!
                })
                transitionBatch.transitionLabels = Tensor<Int32>((0..<stateWithCorrects.count).map({ indexInBatch in
                    return Int32(stateWithCorrects[indexInBatch].corrects.max(by: { guesses[indexInBatch][$0.rawValue] < guesses[indexInBatch][$1.rawValue] })!.rawValue)
                }))
                
                let loss = updateModel(optimizer: optimizer, batch: transitionBatch)
                let logits = model(transitionBatch.features)
                epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: transitionBatch.transitionLabels)
                epochLoss += loss.scalarized()
                batchCount += 1
                
                // apply transition to state
                let applyStates = {
                    for i in 0 ..< stateWithCorrects.count {
                        group.enter()
                        queue.async {
                            let (stateIdx, _) = stateWithCorrects[i]
                            states[stateIdx]!.apply(transition: self.explore(currentEpoch: epoch, guess: bestGuesses[i], correct: Transition(rawValue: Int(transitionBatch.transitionLabels[i].scalar!))!))
                            group.leave()
                        }
                    }
                    group.wait()
                }
                applyStates()
                
                // update working to remove ignoredIndices and new terminal states batch
                let updateWorkingBatch = { [tagger] in
                    for idx in 0..<workingExamples.count {
                        group.enter()
                        queue.async {
                            let shouldUseNewExample = ignoredStateIndices.contains(idx) || (states[idx]?.isTerminal ?? true)
                            if shouldUseNewExample {
                                let newExample = shuffledIterator.next()
                                workingExamples[idx] = newExample
                                states[idx] = newExample != nil ? ParserAutomata(tagger: tagger, rootPrefix: Parser.rootPrefix, sentence: newExample!.sentence) : nil
                            }
                            group.leave()
                        }
                    }
                    group.wait()
                }
                updateWorkingBatch()
                
                // Set new nonTerminalStateIndices
                nonTerminalStatesIndices = self.nonTerminalStateIndices(for: states)
            } while (nonTerminalStatesIndices.count > 0)
            
            // track some metrics
            epochAccuracy /= Float(batchCount)
            epochLoss /= Float(batchCount)
            trainAccuracyResults.append(epochAccuracy)
            trainLossResults.append(epochLoss)
            print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy), Duration: \(Date().timeIntervalSince(epochStartDate)/3600) hours")
            
            // save the model at this checkpoint
            print("Saving model checkpoint after \(epoch) epoch")
            try! serializer.save(model: model, to: TFParseTrainer.savedModelName(epoch: epoch))
            print("trained model saved")
        }
        
        return (trainAccuracyResults, trainLossResults)
    }
    
    private func nonTerminalStateIndices(for states: [ParserAutomata?]) -> [Int] {
        return (0..<states.count).compactMap({
            (states[$0]?.isTerminal ?? true) ? nil : $0
        })
    }
    
    private func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
        return Tensor<Float>(predictions .== truths).mean().scalarized()
    }
    
    @discardableResult private func updateModel(optimizer: Adam<TFParserModel>, batch: TransitionBatch) -> Tensor<Float> {
        let (loss, grad) = model.valueWithGradient { (model: TFParserModel) -> Tensor<Float> in
            let logits = model(batch.features)
            return softmaxCrossEntropy(logits: logits, labels: batch.transitionLabels)
        }
        
        optimizer.update(&model.allDifferentiableVariables, along: grad)
        return loss
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

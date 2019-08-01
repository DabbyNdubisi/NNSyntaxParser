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
    private lazy var optimizer = {
        return Adam(for: model, learningRate: 0.1)
    }()
    
    init(serializer: ModelSerializer, explorationEpochThreshold: Int, explorationProbability: Float, featureProvider: TransitionFeatureProvider, model: TFParserModel) {
        self.serializer = serializer
        self.explorationEpochThreshold = explorationEpochThreshold
        self.explorationProbability = explorationProbability
        self.featureProvider = featureProvider
        self.model = model
    }
    
    func fasterTrain(examples: [ParseExample], batchSize: Int, startEpoch: Int = 1, epochs: Int) {
        guard startEpoch <= epochs else {
            return
        }
        
        let stateGenerator: (String) -> ParserAutomata = { [tagger] sentence in
            let rootPrefixRange = sentence.range(of: UDReader.rootPrefix)!
            let range = sentence.range(of: sentence.suffix(from: rootPrefixRange.upperBound))!
            tagger.string = sentence
            
            let rootToken = Token(i: 0, sentenceRange: rootPrefixRange, posTag: .root)
            return ParserAutomata(
                rootToken: rootToken,
                buffer: tagger.tags(in: range,
                                    unit: .word,
                                    scheme: .lexicalClass,
                                    options: [.omitWhitespace])
                    .enumerated()
                    .map({ Token(i: $0.offset + 1, sentenceRange: $0.element.1, posTag: POSTag(nlTag: $0.element.0!)! ) })
            )
        }
        
        let optimizer = Adam(for: model, learningRate: 0.01)
        for epoch in startEpoch...epochs {
            var epochLoss: Float = 0
            var epochAccuracy: Float = 0
            var ignoredSentences: Int = 0
            let epochStartDate = Date()
            
            let shuffled = examples.shuffled()
            var shuffledIterator = shuffled.makeIterator()
            
            var workingExamples = (0..<batchSize).map({ _ in shuffledIterator.next() })
            var states: [ParserAutomata?] = workingExamples.map({ $0 != nil ? stateGenerator($0!.sentence) : nil })
            var nonTerminalStatesIndices = (0..<states.count).compactMap({ (states[$0] != nil && states[$0]!.isTerminal) ? nil : $0 })
            
            repeat {
                let unsanitizedCorrects: [[Transition]] = {
                    var corrects = [[Transition]](repeating: [], count: nonTerminalStatesIndices.count)
                    let queue = DispatchQueue(label: "queue.corrects")
                    let group = DispatchGroup()
                    for i in 0..<nonTerminalStatesIndices.count {
                        group.enter()
                        queue.async {
                            corrects[i] = states[nonTerminalStatesIndices[i]]!.correctTransition(goldArcs: workingExamples[nonTerminalStatesIndices[i]]!.goldArcs)
                            group.leave()
                        }
                    }
                    _ = group.wait(timeout: .distantFuture)
                    return corrects
                }()
                
                let shouldIgnore = Set<Int>((0..<unsanitizedCorrects.count).filter({ unsanitizedCorrects[$0].isEmpty }))
                if !shouldIgnore.isEmpty {
                    // If there are no correct transitions, it means that the gold tree is not reachable
                    // via our transition system. In such scenarios, it doesn't make sense to continue
                    // training on the current example as it could confuse the model more.
                    ignoredSentences += shouldIgnore.count
                    print("Ignoring \(shouldIgnore.count) sentence as gold tree cannot be attained")
                    print("*** \(ignoredSentences) SENTENCES IGNORED SO FAR")
                }
                
                let stateWithCorrects = (0..<nonTerminalStatesIndices.count).compactMap({ shouldIgnore.contains($0) ? nil : (stateIdx: nonTerminalStatesIndices[$0], corrects: unsanitizedCorrects[$0]) })
                
                var transitionBatch: TransitionBatch = {
                    var feats = [[Int32]](repeating: [], count: stateWithCorrects.count)
                    let queue = DispatchQueue(label: "queue.feats")
                    let group = DispatchGroup()
                    for i in 0..<stateWithCorrects.count {
                        group.enter()
                        queue.async {
                            feats[i] = self.featureProvider.features(for: states[stateWithCorrects[i].stateIdx]!, sentence: workingExamples[stateWithCorrects[i].stateIdx]!.sentence)
                            group.leave()
                        }
                    }
                    _ = group.wait(timeout: .distantFuture)
                    
                    let embeddingInput = EmbeddingInput(indices: Tensor<Int32>(ShapedArray<Int32>(shape: [stateWithCorrects.count, feats[0].count], scalars: feats.flatMap({ $0 }))))
                    return TransitionBatch(
                        features: embeddingInput,
                        transitionLabels: Tensor<Int32>([0])
                    )
                }()
                
                let valids: [[Transition]] = {
                    var valids = [[Transition]](repeating: [], count: stateWithCorrects.count)
                    let queue = DispatchQueue(label: "queue.valids")
                    let group = DispatchGroup()
                    for i in 0..<stateWithCorrects.count {
                        group.enter()
                        queue.async {
                            valids[i] = states[stateWithCorrects[i].stateIdx]!.validTransitions()
                            group.leave()
                        }
                    }
                    _ = group.wait(timeout: .distantFuture)
                    return valids
                }()
                
                let guesses = model(transitionBatch.features) // predictions
                
                let bestGuesses = (0..<valids.count).map({ batchIdx in
                    valids[batchIdx].max(by: { guesses[batchIdx][$0.rawValue] < guesses[batchIdx][$1.rawValue] })!
                })
                transitionBatch.transitionLabels = Tensor<Int32>((0..<stateWithCorrects.count).map({ batchIdx in
                    Int32(stateWithCorrects[batchIdx].corrects.max(by: { guesses[batchIdx][$0.rawValue] < guesses[batchIdx][$1.rawValue] })!.rawValue)
                }))
                
                let loss = updateModel(optimizer: optimizer, batch: transitionBatch)
                let logits = model(transitionBatch.features)
                epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: transitionBatch.transitionLabels)
                epochLoss += loss.scalarized()
                
                let applyStates = {
                    let queue = DispatchQueue(label: "queue.apply_states")
                    let group = DispatchGroup()
                    for i in 0 ..< stateWithCorrects.count {
                        group.enter()
                        queue.async {
                            let (stateIdx, _) = stateWithCorrects[i]
                            states[stateIdx]!.apply(transition: self.explore(currentEpoch: epoch, guess: bestGuesses[i], correct: Transition(rawValue: Int(transitionBatch.transitionLabels[i].scalar!))!))
                            group.leave()
                        }
                    }
                    _ = group.wait(timeout: .distantFuture)
                }
                applyStates()
                
                // update
                let ignoredStatesIndices = shouldIgnore.map({ nonTerminalStatesIndices[$0] })
                (0..<workingExamples.count).forEach { idx in
                    let shouldUseNewExample = ignoredStatesIndices.contains(idx) || (states[idx]?.isTerminal ?? true)
                    if shouldUseNewExample {
                        let newExample = shuffledIterator.next()
                        workingExamples[idx] = newExample
                        states[idx] = newExample != nil ? stateGenerator(newExample!.sentence) : nil
                    }
                }
                nonTerminalStatesIndices = (0..<states.count).compactMap({
                    (states[$0] == nil || (states[$0]?.isTerminal ?? true) || ignoredStatesIndices.contains($0)) ? nil : $0
                })
            } while (nonTerminalStatesIndices.count > 0)
            
            print("Epoch \(epoch): Loss: \(epochLoss), Duration: \(Date().timeIntervalSince(epochStartDate)/3600) hours")
            print("Saving model checkpoint after \(epoch) epoch")
            try! serializer.save(model: model, to: TFParseTrainer.savedModelName(epoch: epoch))
            print("trained model saved")
        }
    }
    
    func train(examples: [ParseExample], epochs: Int) {
        let optimizer = Adam(for: model, learningRate: 0.01)
        
        for epoch in 1...epochs {
            var epochLoss: Float = 0
            var epochAccuracy: Float = 0
            var tokenCount: Int = 0
            var ignoredSentences: Int = 0
            let epochStartDate = Date()
            
            let shuffled = examples.shuffled()
            for i in 0..<shuffled.count {
                let (sentence, goldArcs) = shuffled[i]
                let rootPrefixRange = sentence.range(of: UDReader.rootPrefix)!
                let range = sentence.range(of: sentence.suffix(from: rootPrefixRange.upperBound))!
                tagger.string = sentence
                
                let rootToken = Token(i: 0, sentenceRange: rootPrefixRange, posTag: .root)
                var state = ParserAutomata(
                    rootToken: rootToken,
                    buffer: tagger.tags(in: range,
                                        unit: .word,
                                        scheme: .lexicalClass,
                                        options: [.omitWhitespace])
                        .enumerated()
                        .map({ Token(i: $0.offset + 1, sentenceRange: $0.element.1, posTag: POSTag(nlTag: $0.element.0!)! ) })
                )
                
                while !state.isTerminal  {
                    let corrects = state.correctTransition(goldArcs: goldArcs)
                    guard corrects.count > 0 else {
                        // If there are no correct transitions, it means that the gold tree is not reachable
                        // via our transition system. In such scenarios, it doesn't make sense to continue
                        // training on the current example as it could confuse the model more.
                        ignoredSentences += 1
                        print("Ignoring sentence as gold tree cannot be attained: \(sentence)")
                        print("*** \(ignoredSentences) SENTENCES IGNORED SO FAR")
                        break
                    }
                    
                    let feats = featureProvider.features(for: state, sentence: sentence)
                    let features = EmbeddingInput(indices: Tensor<Int32>(ShapedArray<Int32>(shape: [1, feats.count], scalars: feats)))
                    let valids = state.validTransitions()
                    let guesses = model(features) // predictions
                    
                    let bestGuess = valids.max(by: { guesses[0][$0.rawValue] < guesses[0][$1.rawValue] })!
                    let correctTransition = corrects.max(by: { guesses[0][$0.rawValue] < guesses[0][$1.rawValue] })!
                    let loss = updateModel(optimizer: optimizer, features: features, correct: correctTransition)
                    
                    let logits = model(features)
                    epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: Tensor<Int32>([Int32(correctTransition.rawValue)]))
                    epochLoss += loss.scalarized()
                    tokenCount += 1
                    
                    state.apply(transition: explore(currentEpoch: epoch, guess: bestGuess, correct: correctTransition))
                }
            }
            
            epochAccuracy /= Float(tokenCount)
            epochLoss /= Float(tokenCount)
            print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy), Duration: \(Date().timeIntervalSince(epochStartDate)/3600) hours")
            
            print("Saving model checkpoint after \(epoch) epoch")
            try! serializer.save(model: model, to: "\(Constants.trainModelCheckpointName)_\(epoch)epoch")
            print("trained model saved")
        }
    }
    
    private func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
        return Tensor<Float>(predictions .== truths).mean().scalarized()
    }
    
    @discardableResult private func updateModel(optimizer: Adam<TFParserModel>, features: EmbeddingInput, correct: Transition) -> Tensor<Float> {
        return updateModel(optimizer: optimizer, batch: TransitionBatch(features: features, transitionLabels: Tensor<Int32>(Int32(correct.rawValue))))
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

// MARK: - Dynamic oracle
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

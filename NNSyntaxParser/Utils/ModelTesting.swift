//
//  ModelTesting.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-08-04.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow

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

func test(parser: Parser, examples: [ParseExample]) -> (las: Float, uas: Float) {
    var lasCorrects = 0
    var uasCorrects = 0
    var totals = 0
    for example in examples {
        let answer = try! parser.parse(sentence: example.sentence)
        lasCorrects += answer.heads.enumerated().filter({ $0.element?.head == example.goldArcs[$0.offset]?.head && $0.element?.relationship == example.goldArcs[$0.offset]?.relationship }).count
        uasCorrects += answer.heads.enumerated().filter({ $0.element?.head == example.goldArcs[$0.offset]?.head }).count
        totals += answer.heads.count
    }

    return (Float(lasCorrects)/Float(totals), Float(uasCorrects)/Float(totals))
}

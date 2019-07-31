//
//  EmbeddingsReader.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-11.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow
import Python

struct EmbeddingsReader {
    static func dependencyEmbedding(from wordEmbeddingsURL: URL) -> DependencyEmbedding {
        let np = Python.import("numpy")
        
        let words = np.loadtxt(
            wordEmbeddingsURL.path,
            comments: "None",
            usecols: [0],
            dtype: "str",
            encoding: "utf-8")
        let index2Words = [String](words)!
        
        let npWordEmbeddings = np.loadtxt(
            wordEmbeddingsURL.path,
            comments: "None",
            usecols: [Int](1...50), // embedding size is 50
            dtype: Float.numpyScalarTypes.first!,
            encoding: "utf-8"
        )
        let wordEmbeddings = Tensor<Float>(numpy: npWordEmbeddings)!
        
        return DependencyEmbedding(index2Word: index2Words, wordEmbeddings: wordEmbeddings)
    }
}

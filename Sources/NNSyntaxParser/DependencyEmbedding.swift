//
//  DependencyEmbedding.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-12.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow
import NaturalLanguage
import LanguageParseModels

struct DependencyEmbedding {
    let featureProvider: TransitionFeatureProvider
    /// consolidated embeddings array
    let embedding: Tensor<Float>
    
    init(index2Word: [String], wordEmbeddings: Tensor<Float>) {
        featureProvider = TransitionFeatureProvider(index2Word: index2Word)
        let embeddingDimension = wordEmbeddings.shape[1]
        let tokenEmbeddings = Tensor<Float>.init(randomUniform: [TransitionFeatureProvider.domainTokens.count, embeddingDimension])
        
        // add wordEmbeddings
        self.embedding = tokenEmbeddings.concatenated(with: wordEmbeddings)
    }
}

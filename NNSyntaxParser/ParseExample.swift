//
//  ParseExample.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-14.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation

/// sentence and expected dependency arcs
/// for a single example
typealias ParseExample = (sentence: String, goldArcs: [Dependency?])

//
//  DataRequirement.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-20.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation

/// Data that needs to be downloaded
enum DataRequirement: CaseIterable {
    // Glove (Wikipedia + Gigaword5) word embeddings
    case glove
    // UD tree bank for model training
    case udTreebank
    
    var file: RequirementFile {
        switch self {
        case .glove:
            return RequirementFile(name: "glove", extension: .zip, url: URL(string: "http://nlp.stanford.edu/data/glove.6B.zip")!)
        case .udTreebank:
            return RequirementFile(name: "UD_treebank_deps", extension: .tgz, url: URL(string: "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz?sequence=4&isAllowed=y")!)
        }
    }
}

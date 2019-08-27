//
//  Constants.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-10.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation

enum Constants {
    static let folderName = "nn_syntax_parser"
    static var defaultLocation: URL = {
        let directoryURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(Constants.folderName)
        if !FileManager.default.fileExists(atPath: directoryURL.path) {
            try! FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: false)
        }
        return directoryURL
    }()
}

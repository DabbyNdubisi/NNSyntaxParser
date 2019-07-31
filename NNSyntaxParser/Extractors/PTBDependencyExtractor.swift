//
//  PTBDependencyExtractor.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-10.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import Python

/// Extractor for the Penn Treebank dependency (PTB).
struct PTBDependencyExtractor {
    /// The supported file type for the extractor
    var supportedExtension: FileExtension {
        .zip
    }
    
    var extractedFileName: String {
        return "PTB_deps"
    }
    
    func extract(from fileName: String, into destination: URL = Constants.defaultLocation) {
        let sourceLocation = destination
            .appendingPathComponent(fileName)
            .appendingPathExtension(supportedExtension.rawValue)
        let extractedDestination = destination
            .appendingPathComponent(extractedFileName, isDirectory: true)
        
        if FileManager.default.fileExists(atPath: extractedDestination.path) {
            try! FileManager.default.removeItem(at: extractedDestination)
        }
        try! FileManager.default.createDirectory(at: extractedDestination, withIntermediateDirectories: false)
        
        let zf = Python.import("zipfile")
        let file = zf.ZipFile(sourceLocation.path)
        let toBeExtracted: [PythonObject] = file.infolist().filter {
            String($0.filename)!.contains("wsj")
        }.map {
            let nameString = String($0.filename)!
            let range = nameString.range(of: "wsj")!
            $0.filename = PythonObject(String(nameString.suffix(from: range.lowerBound)))
            return $0
        }
        file.extractall(extractedDestination.path, toBeExtracted)
        file.close()
    }
}



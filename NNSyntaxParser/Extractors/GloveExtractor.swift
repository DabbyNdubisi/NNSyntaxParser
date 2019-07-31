//
//  GloveExtractor.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-10.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import Python

/// Extractor for the Glove embeddings.
/// Extracts the 50-dimensional word embeddings
struct GloveExtractor {
    static var extractedResourcePath: URL {
        return Constants.defaultLocation
            .appendingPathComponent("glove")
            .appendingPathComponent("glove.6B.50d")
            .appendingPathExtension("txt")
    }
    
    /// The supported file type for the extractor
    private var supportedExtension: FileExtension {
        .zip
    }
    
    private var extractedFileName: String {
        return "glove"
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
        file.extractall(extractedDestination.path, ["glove.6B.50d.txt"])
        file.close()
    }
}

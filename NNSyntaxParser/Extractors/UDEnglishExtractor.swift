//
//  UDEnglishExtractor.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-10.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import Python

/// Extractor for the Universal Dependency (UD) treeebank.
/// Extracts the UDs for English `UD_English`
struct UDEnglishExtractor {
    static var trainDataURL: URL {
        return Constants.defaultLocation
            .appendingPathComponent("UD_English")
            .appendingPathComponent("en_ewt-ud-train")
            .appendingPathExtension("conllu")
    }
    
    static var testDataURL: URL {
        return Constants.defaultLocation
            .appendingPathComponent("UD_English")
            .appendingPathComponent("en_ewt-ud-test")
            .appendingPathExtension("conllu")
    }
    
    static var validationDataURL: URL {
        return Constants.defaultLocation
            .appendingPathComponent("UD_English")
            .appendingPathComponent("en_ewt-ud-dev")
            .appendingPathExtension("conllu")
    }
    
    /// The supported file type for the extractor
    var supportedExtension: FileExtension {
        .tgz
    }
    
    var extractedFileName: String {
        return "UD_English"
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
        
        // only train on EWT for now
        let directoryComponentName = "UD_English-EWT/"
        let tf = Python.import("tarfile")
        let tarFile = tf.open(sourceLocation.path)
        let toBeExtracted: [PythonObject] = tarFile.getmembers().filter {
            String($0.name)!.contains(directoryComponentName)
        }.map {
            let nameString = String($0.name)!
            let range = nameString.range(of: directoryComponentName)!
            $0.name = PythonObject(String(nameString.suffix(from: range.upperBound)))
            return $0
        }
        tarFile.extractall("\(extractedDestination.path)", toBeExtracted)
        tarFile.close()
    }
}

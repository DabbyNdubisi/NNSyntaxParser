//
//  ModelSaver.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-21.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import TensorFlow

struct ModelSerializer {
    private let location: URL
    
    init(location: URL = Constants.defaultLocation) {
        self.location = location
    }
    
    func save(model: TFParserModel, to name: String) throws {
        let encodedData = try JSONEncoder().encode(model)
        try encodedData.write(to: location.appendingPathComponent(name), options: .atomic)
    }
    
    func loadModel(name: String) throws -> TFParserModel {
        let decodedData = try Data(contentsOf: location.appendingPathComponent(name))
        return try JSONDecoder().decode(TFParserModel.self, from: decodedData)
    }
    
    func modelExists(name: String) -> Bool {
        return FileManager.default.fileExists(atPath: location.appendingPathComponent(name).path)
    }
}

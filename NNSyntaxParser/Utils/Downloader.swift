//
//  Downloader.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-20.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation

/// A basic data downloader
struct Downloader {
    let session = URLSession(configuration: .default)
    
    func download(from source: URL, to name: String, extension: FileExtension) {
        let destination = Constants.defaultLocation
            .appendingPathComponent(name)
            .appendingPathExtension(`extension`.rawValue)
        
        let semaphore = DispatchSemaphore(value: 0)

        var maybeData: Data?
        var error: Error?
        let dataTask = session.dataTask(with: source, completionHandler: {
            maybeData = $0
            error = $2
            semaphore.signal()
        })
        dataTask.resume()
        _ = semaphore.wait(timeout: .distantFuture)
        
        guard let data = maybeData else {
            print("Unable to download file: \(name) with error: \(String(describing: error))")
            return
        }
        try! data.write(to: destination, options: .atomic)
    }
    
    func download(requirement: RequirementFile) {
        download(from: requirement.url, to: requirement.name, extension: requirement.extension)
    }
}

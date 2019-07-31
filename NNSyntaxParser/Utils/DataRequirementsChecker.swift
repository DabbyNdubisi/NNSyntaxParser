//
//  DataRequirementsChecker.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-20.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation

/// DataRequirementsChecker error
enum DataRequirementsCheckerError: Error {
    case requirementsNotMet(unmetRequirements: [DataRequirement])
}

/// Checks that Data requirements are satisfied
struct DataRequirementsChecker {
    func checkRequirements() throws {
        let downloader = Downloader()
        let unmetRequirements = DataRequirement.allCases.filter {
            !downloader.requirementAlreadyDownloaded($0.file)
        }
        
        guard unmetRequirements.count == 0 else {
            throw DataRequirementsCheckerError.requirementsNotMet(unmetRequirements: unmetRequirements)
        }
    }
}

private extension Downloader {
    func requirementAlreadyDownloaded(_ file: RequirementFile) -> Bool {
        let expectedURL = Constants.defaultLocation
            .appendingPathComponent(file.name)
            .appendingPathExtension(file.extension.rawValue)
        return FileManager.default.fileExists(atPath: expectedURL.path)
    }
}

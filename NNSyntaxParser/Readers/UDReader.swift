//
//  UDReader.swift
//  NNSyntaxParser
//
//  Created by Dabeluchi Ndubisi on 2019-07-12.
//  Copyright © 2019 Dabeluchi Ndubisi. All rights reserved.
//

import Foundation
import Python
import NaturalLanguage

struct UDReader {
    private static var rootPrefix: String {
        return Parser.rootPrefix
    }
    
    static func readTrainData() -> [ParseExample] {
        let readTokens = readSentenceTokens(from: UDEnglishExtractor.trainDataURL)
        return generateExamples(from: readTokens)
    }
    
    static func readValidationData() -> [ParseExample] {
        let readTokens = readSentenceTokens(from: UDEnglishExtractor.validationDataURL)
        return generateExamples(from: readTokens)
    }
    
    static func readTestData() -> [ParseExample] {
        let readTokens = readSentenceTokens(from: UDEnglishExtractor.testDataURL)
        return generateExamples(from: readTokens)
    }
    
    // MARK: - Private helpers
    private static func generateExamples(from data: [[String]]) -> [ParseExample] {
        var sentences = [String]()
        var sent2TokenData = [Int: [UDTokenData]]()
        var currentSentence = "\(rootPrefix) "
        var tokenBuff = [UDTokenData]()
        
        let markSentence: () -> Void = {
            sentences.append(currentSentence)
            sent2TokenData[sentences.count-1] = tokenBuff
            currentSentence = "\(rootPrefix) "
            tokenBuff = []
        }
        
        var i = 0
        var prevTokenData: UDTokenData?
        while true {
            guard i < data.count else {
                markSentence()
                break
            }
            
            guard !data[i][0].contains(".1") else {
                // tokens with id `x.1` added on parse to
                // potentially help with POS tagging.
                // They have no dependency information, so we can ignore
                i += 1
                continue
            }
            
            let tokenData = UDTokenData(raw: data[i])!
            if tokenData.isFirstTokenForNewSentence && prevTokenData != nil {
                markSentence()
            }
            
            // Add space iff previous token wants space after and
            // the current token isn't the first token of a new sentence
            if (prevTokenData?.spaceAfter ?? false) && !tokenData.isFirstTokenForNewSentence {
                currentSentence += " "
            }
            
            currentSentence += tokenData.form
            tokenBuff.append(tokenData)
            prevTokenData = tokenData
            i += 1
        }
        
        let corruptedIndices = Set(corruptedSentences(sentences: sentences, corpusSent2Tokens: sent2TokenData))
        return sentences.lazy
            .enumerated()
            .filter({ !corruptedIndices.contains($0.offset) })
            .map({ offset, element in
                (
                    element,
                    // first element is `nil` because <ROOT> has no dependency
                    [nil] + sent2TokenData[offset]!.map({ Dependency(head: $0.head, relationship: $0.dependencyRelation) })
                )
            })
            .filter({ !isDependencyNonProjective(goldArcs: $0.goldArcs) })
    }
    
    private static func isDependencyNonProjective(goldArcs: [Dependency?]) -> Bool {
        // Roughly 477 sentences have non-projective dependencies.
        // We need to remove these.
        let enumeratedArcs = goldArcs.enumerated()
        for a in enumeratedArcs {
            guard let head1 = a.element?.head else {
                continue
            }
            
            for b in enumeratedArcs {
                guard let head2 = b.element?.head else {
                    continue
                }
                
                if head1 == 0 || head2 == 0 {
                    continue
                } else if (a.offset > head2 && a.offset < b.offset && head1 < head2) || (a.offset < head2 && a.offset > b.offset && head1 < b.offset) {
                    return true
                } else if a.offset < head1 && head1 != head2 {
                    if (head1 > head2 && head1 < b.offset && a.offset < head2) || (head1 < head2 && head1 > b.offset && a.offset < b.offset) {
                        return true
                    }
                }
            }
        }
        return false
    }
    
    /// returns the indices of the sentences where Apple's NLTagger tokenization scheme doesn't
    /// match with the UD tokenization format
    private static func corruptedSentences(sentences: [String], corpusSent2Tokens: [Int: [UDTokenData]]) -> [Int] {
        var appleSent2Tokens: [Int: [String]] = [:]
        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        for i in 0..<sentences.count {
            let string = sentences[i]
            let rootPrefixRange = sentences[i].range(of: rootPrefix)!
            let range = string.range(of: string.suffix(from: rootPrefixRange.upperBound))!
            tagger.string = string
            appleSent2Tokens[i] = tagger.tags(in: range, unit: .word, scheme: .lexicalClass, options: [.omitWhitespace]).map({ String(string[$0.1]) })
        }
        
        // 1766/12543
        // 1797/12543 ~ 15% corruption train discrepancies
        // 315/2077 ~ 15% corruption test discrepancies
        //
        // token count and order must match to be considered valid
        return sentences.lazy.enumerated()
            .filter({ corpusSent2Tokens[$0.offset]?.map({ $0.form }) != appleSent2Tokens[$0.offset] })
            .map({ $0.offset })
    }
    
    private static func readSentenceTokens(from url: URL) -> [[String]] {
        let np = Python.import("numpy")
        let tokens = np.loadtxt(
            url.path,
            comments: ["# newdoc", "# sent_id", "# text"],
            dtype: "str",
            encoding: "utf-8")
        return [[String]](tokens)!
    }
}

// MARK: - UDTokenData
extension UDReader {
    private struct UDTokenData {
        /// Word index, integer starting at 1 for each new sentence; may be a range for tokens with multiple words.
        let id: Int
        /// Word form or punctuation symbol.
        let form: String
        /// Lemma or stem of word form.
        let lemma: String
        /// Universal part-of-speech tag drawn from our revised version of the Google universal POS tags.
        let uPOSTag: String
        /// Language-specific part-of-speech tag; underscore if not available.
        let xPOSTag: String
        /// List of morphological features from the universal feature inventory or
        /// from a defined language-specific extension; underscore if not available.
        let feats: String
        /// Head of the current token, which is either a value of ID or zero (0).
        let head: Int
        /// Universal Stanford dependency relation to the HEAD (root iff HEAD = 0)
        /// or a defined language-specific subtype of one.
        let dependencyRelation: DependencyRelation
        /// List of secondary dependencies (head-deprel pairs).
        let deps: String
        /// Any other annotation.
        let misc: String
        
        var spaceAfter: Bool {
            return !misc.contains("SpaceAfter=No")
        }
        
        var isFirstTokenForNewSentence: Bool {
            return id == 1
        }
        
        init?(raw: [String]) {
            guard raw.count == 10 else { return nil }
            
            id = Int(raw[0])!
            form = raw[1]
            lemma = raw[2]
            uPOSTag = raw[3]
            xPOSTag = raw[4]
            feats = raw[5]
            head = Int(raw[6])!
            let relation = String(raw[7].split(separator: ":").first!)
            dependencyRelation = DependencyRelation(rawValue: relation)!
            deps = raw[8]
            misc = raw[9]
        }
    }
}

// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: DataStructures.proto
//
// For information on using the generated types, please see the documenation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that your are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

///*
/// A mapping from a string
/// to a 64-bit integer.
struct CoreML_Specification_StringToInt64Map {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var map: Dictionary<String,Int64> = [:]

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A mapping from a 64-bit integer
/// to a string.
struct CoreML_Specification_Int64ToStringMap {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var map: Dictionary<Int64,String> = [:]

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A mapping from a string
/// to a double-precision floating point number.
struct CoreML_Specification_StringToDoubleMap {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var map: Dictionary<String,Double> = [:]

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A mapping from a 64-bit integer
/// to a double-precision floating point number.
struct CoreML_Specification_Int64ToDoubleMap {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var map: Dictionary<Int64,Double> = [:]

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A vector of strings.
struct CoreML_Specification_StringVector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var vector: [String] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A vector of 64-bit integers.
struct CoreML_Specification_Int64Vector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var vector: [Int64] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A vector of floating point numbers.
struct CoreML_Specification_FloatVector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var vector: [Float] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A vector of double-precision floating point numbers.
struct CoreML_Specification_DoubleVector {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var vector: [Double] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A range of int64 values
struct CoreML_Specification_Int64Range {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var minValue: Int64 = 0

  var maxValue: Int64 = 0

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A set of int64 values
struct CoreML_Specification_Int64Set {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var values: [Int64] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// A range of double values
struct CoreML_Specification_DoubleRange {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var minValue: Double = 0

  var maxValue: Double = 0

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension CoreML_Specification_StringToInt64Map: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".StringToInt64Map"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufInt64>.self, value: &self.map)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufInt64>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_StringToInt64Map, rhs: CoreML_Specification_StringToInt64Map) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_Int64ToStringMap: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".Int64ToStringMap"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufString>.self, value: &self.map)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufString>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_Int64ToStringMap, rhs: CoreML_Specification_Int64ToStringMap) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_StringToDoubleMap: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".StringToDoubleMap"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufDouble>.self, value: &self.map)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufString,SwiftProtobuf.ProtobufDouble>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_StringToDoubleMap, rhs: CoreML_Specification_StringToDoubleMap) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_Int64ToDoubleMap: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".Int64ToDoubleMap"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "map"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufDouble>.self, value: &self.map)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.map.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufInt64,SwiftProtobuf.ProtobufDouble>.self, value: self.map, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_Int64ToDoubleMap, rhs: CoreML_Specification_Int64ToDoubleMap) -> Bool {
    if lhs.map != rhs.map {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_StringVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".StringVector"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedStringField(value: &self.vector)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitRepeatedStringField(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_StringVector, rhs: CoreML_Specification_StringVector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_Int64Vector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".Int64Vector"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedInt64Field(value: &self.vector)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitPackedInt64Field(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_Int64Vector, rhs: CoreML_Specification_Int64Vector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_FloatVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".FloatVector"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedFloatField(value: &self.vector)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitPackedFloatField(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_FloatVector, rhs: CoreML_Specification_FloatVector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_DoubleVector: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".DoubleVector"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "vector"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedDoubleField(value: &self.vector)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.vector.isEmpty {
      try visitor.visitPackedDoubleField(value: self.vector, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_DoubleVector, rhs: CoreML_Specification_DoubleVector) -> Bool {
    if lhs.vector != rhs.vector {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_Int64Range: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".Int64Range"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "minValue"),
    2: .same(proto: "maxValue"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularInt64Field(value: &self.minValue)
      case 2: try decoder.decodeSingularInt64Field(value: &self.maxValue)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.minValue != 0 {
      try visitor.visitSingularInt64Field(value: self.minValue, fieldNumber: 1)
    }
    if self.maxValue != 0 {
      try visitor.visitSingularInt64Field(value: self.maxValue, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_Int64Range, rhs: CoreML_Specification_Int64Range) -> Bool {
    if lhs.minValue != rhs.minValue {return false}
    if lhs.maxValue != rhs.maxValue {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_Int64Set: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".Int64Set"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "values"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedInt64Field(value: &self.values)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.values.isEmpty {
      try visitor.visitPackedInt64Field(value: self.values, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_Int64Set, rhs: CoreML_Specification_Int64Set) -> Bool {
    if lhs.values != rhs.values {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_DoubleRange: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".DoubleRange"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "minValue"),
    2: .same(proto: "maxValue"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularDoubleField(value: &self.minValue)
      case 2: try decoder.decodeSingularDoubleField(value: &self.maxValue)
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.minValue != 0 {
      try visitor.visitSingularDoubleField(value: self.minValue, fieldNumber: 1)
    }
    if self.maxValue != 0 {
      try visitor.visitSingularDoubleField(value: self.maxValue, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_DoubleRange, rhs: CoreML_Specification_DoubleRange) -> Bool {
    if lhs.minValue != rhs.minValue {return false}
    if lhs.maxValue != rhs.maxValue {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

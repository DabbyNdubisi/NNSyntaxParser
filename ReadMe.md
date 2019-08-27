A swift project which trains a model for parsing English using Swift for Tensorflow.

## Setup Instructions
This project depends on Swift for Tensorflow. You can install the Swift for Tensorflow toolchain from [tensorflow/swift](https://github.com/tensorflow/swift/blob/master/Installation.md)

### Constraints
- This project has to run with the `Legacy Build System` in order for Swift for Tensorflow to work. SwiftPM functionality on XCode 11 requires the `New Build System`, so SwiftPM has to be used from the command line to manage dependencies
- This project disables Library Validation as Swift for Tensorflow framework isn't code signed

To setup the project run:
```
$swift package update

[Optional Step to regenerate Xcode project]
$swift package generate-xcodeproj
```

## Execution instructions
- Ensure that Xcode is using the Swift for Tensorflow toolchain.
- Ensure that the project is using the `Legacy Build System`

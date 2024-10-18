# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2024-10-03
### Plan
The Plan is making modular core. So when training in kaggle only pull and make some config for dataset (kaggle) and training / inference using only script with args.

## [Unreleased] - 2024-10-18
Build up the app so it can run demo. after that, find out the model is ugly when we tested. so maybe i back to training again.

idk what happen, but when i see the result of training by inspecting image (visualize generate), seems fine. but in interface, run badly.

## [0.0.1] - 2024-10-19
After looking on dataset, the problem before is because we training the image in shape 1024x1024 Close up Face Image, so when retrive image with face and a bit body can make model mess up. so require image like example provided to make some nice result.
### Feature:
- Turn Image of Face Close up into a Comic Style.

### Changed
- The Example is change, so user will get some insiration for the input
- 2 Output, first is the original image after transformation, and second is image after sending to model

### Removed
- Old Example

### Fixed
- When the resize doesnt match 256x256, because not provide in tuple, so resize only height when passed 1 paramters.
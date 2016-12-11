Based on the following paper: ["Continuous control with deep reinforcement learning"](http://arxiv.org/abs/1509.02971) by T. P. Lillicrap et al.

## Installation

Install [Gym](https://github.com/openai/gym#installation), [TensorFlow](https://www.tensorflow.org/get_started/os_setup.html), [MoviePy](http://zulko.github.io/moviepy/install.html) and [protobuf](https://github.com/google/protobuf/releases)

### Compile the needed protocol buffers:

```bash
pip install protobuf  # To install protobuf
brew install protobuf  # To install protoc
protoc -I=protos --python_out=ddpg protos/options.proto
```

## Usage

...

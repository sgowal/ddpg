Based on the following paper: ["Continuous control with deep reinforcement learning"](http://arxiv.org/abs/1509.02971) by T. P. Lillicrap et al.

## Installation

Install [Gym](https://github.com/openai/gym#installation), [TensorFlow](https://www.tensorflow.org/get_started/os_setup.html), [MoviePy](http://zulko.github.io/moviepy/install.html) and [protobuf](https://github.com/google/protobuf/releases)

### Compile the needed protocol buffers:

```bash
pip install protobuf  # To install protobuf
brew install protobuf  # To install protoc on MacOS
sudo apt-get install protobuf-compiler  # To install protoc on Ubuntu
protoc -I=protos --python_out=ddpg protos/options.proto
```

## Usage

```bash
# To list all environments.
python run.py --list

# To search an environment.
python run.py --search CartPole.*

# To run with default options.
python run.py \
  --output_directory path/to/some/experiment \
  --environment=CartPole-v0

# To specify some options.
python run.py \
  --output_directory path/to/some/experiment \
  --environment=CartPole-v0 \
  --options="evaluate_after_timesteps: 2000 device: '/gpu:0'"

# Restore from a previously stored checkpoint.
python run.py \
  --output_directory path/to/some/previously/run/experiment \
  --environment=CartPole-v0 \
  --restore
```

Feel free to run tensorboard while training:

```bash
tensorboard --logdir=path/to/some/experiment
```

## Example of outputs

![Performance](https://github.com/sgowal/ddpg/raw/master/doc/performance.png)
![CartPole-v0](https://github.com/sgowal/ddpg/raw/master/doc/cartpole.gif)

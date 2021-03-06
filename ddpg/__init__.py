from agent import Agent
from core import Start
from manager import AVERAGE_REWARD_TAG
from manager import STDDEV_REWARD_TAG

try:
  from options_pb2 import Options
  from options_pb2 import OptionVariants
except ImportError as e:
  print 'Cannot find options_pb2.py, run "protoc -I=protos --python_out=ddpg protos/*.proto"'
  raise e

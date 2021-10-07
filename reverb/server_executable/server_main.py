# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Re-usable Reverb server that can be packaged as a binary.

It is functionally equivalent to instantiating the reverb.Server
class, but instead of the user's configuration coming from constructor
arguments, it comes from a command-line argument in textproto format.
"""

from absl import app
from absl import flags
from absl import logging
import reverb
from reverb.platform.default import server_main_command_line_args
from reverb.server_executable import server_from_proto

FLAGS = flags.FLAGS


def main(unused_argv):
  config_proto = (
      server_main_command_line_args.get_server_config_proto())
  port = config_proto.port
  table_configs = server_from_proto.tables_from_proto(
      config_proto.tables)
  logging.info('Configuring reverb for %d tables', len(table_configs))
  server = reverb.Server(tables=table_configs, port=port)
  logging.info('Reverb started.')
  server.wait()


# This is used as entry point for the console_script defined in
# setup.py
def app_run_main():
  app.run(main)


if __name__ == '__main__':
  app_run_main()

from argparse import ArgumentParser
import yaml
import sys

HELP = """
In the configuration file, you should specify the following parameters:

device
    The target device to infer on:  CPU, GPU, FPGA or MYRIAD is acceptable. 
mod_fd
    Path to the face detection model (XML file)
mod_lm
    Path to the landmarks model (XML file)
mod_hp
    Path to the head pose estimation model (XML file)
mod_ge
    Path to the gaze estimation model (XML file)
input_type
    Input file type (either `cam` or `video`)
input_file_path
    Input file path (required for `video` input_type)
"""

def validate_args(args):
    """
    - Validate arguments passed from command line.
    - Extract arguments from configuration file.
    """

    # Check if the user wants to print the config file help page
    print_help = args.print
    if print_help:
        print(HELP)
        sys.exit(0)

    # Read the configuration parameters
    config_file = args.config
    with open(config_file, 'r') as f:
        config_text = ''.join(f.readlines())

    parsed_args = yaml.safe_load(config_text)

    print(parsed_args)

    return parsed_args

def get_args():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-cf', '--config', type=str, 
        help="The path to the YAML configuration file.")
    group.add_argument('-p', '--print', action='store_true',
        help="Print the configuration file Help page.")

    args = parser.parse_args()

    params = validate_args(args)
    return params

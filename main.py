from RL_tiago import train
import rclpy
from rclpy.node import Node


def main():

    rclpy.init()

    node = Node('tiago_control')

    try:
        train()

    finally:
        node.destroy_node()
        rclpy.shutdown()
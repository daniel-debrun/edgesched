"""SKETCH: wrap an edgesched Arbiter in a ROS 2 node. Untested; not part of the test suite.

One process owns the accelerator. Each subscription is a stream with its own
period/deadline; callbacks submit to the arbiter and publish from the future's
done-callback, so no executor thread blocks on inference. Deadlines are
measured from the message header stamp, i.e. from when the sensor produced
the data, not from when the callback happened to run.

Assumes rclpy, sensor_msgs and std_msgs are available (e.g. a Humble/Jazzy
environment). ``preprocess`` / ``postprocess`` are placeholders you replace.

    ros2 run <your_pkg> edgesched_node   # after packaging it yourself
"""

from __future__ import annotations

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"ROS 2 Python packages not found ({e}); this example needs rclpy.") from e

import time

from edgesched.config import load_workload
from edgesched.runtime import Arbiter, DeadlineMissed, TorchBackend
from edgesched.zoo import actor_mlp, tiny_cnn


def stamp_to_arbiter_clock(stamp, node: Node) -> float:
    """Map a ROS header stamp onto the arbiter's perf_counter clock."""
    age = (node.get_clock().now().nanoseconds - (stamp.sec * 1_000_000_000 + stamp.nanosec)) * 1e-9
    return time.perf_counter() - max(0.0, age)


class EdgeschedNode(Node):
    def __init__(self) -> None:
        super().__init__("edgesched")
        self.declare_parameter("workload", "workloads/multi_agent_cell.yaml")
        self.declare_parameter("policy", "edf_batch")
        workload = load_workload(self.get_parameter("workload").value)

        backends = {
            "perception_cnn": TorchBackend(tiny_cnn()[0]),
            "actor_mlp": TorchBackend(actor_mlp()[0]),
            "planner": TorchBackend(actor_mlp(hidden=1024)[0]),
        }
        self.arbiter = Arbiter(
            workload.models,
            backends,
            workload.policy(self.get_parameter("policy").value),
            late_policy="drop",
            admission="enforce",
        ).start()
        for task in workload.tasks:
            self.arbiter.register_stream(task)

        self.seg_pub = self.create_publisher(Float32MultiArray, "segmentation", 1)
        self.create_subscription(Image, "camera/image", self.on_image, 1)

    def on_image(self, msg: Image) -> None:
        release = stamp_to_arbiter_clock(msg.header.stamp, self)
        future = self.arbiter.submit("camera", preprocess(msg), release=release)
        future.add_done_callback(self._publish_segmentation)

    def _publish_segmentation(self, future) -> None:
        try:
            out = future.result()
        except DeadlineMissed:
            self.get_logger().debug("camera frame dropped: deadline passed in queue")
            return
        self.seg_pub.publish(postprocess(out))

    def destroy_node(self) -> None:
        self.arbiter.stop(drain=False)
        super().destroy_node()


def preprocess(msg: Image):  # placeholder: decode msg.data into a CHW float tensor
    raise NotImplementedError


def postprocess(tensor) -> Float32MultiArray:  # placeholder
    raise NotImplementedError


def main() -> None:
    rclpy.init()
    node = EdgeschedNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

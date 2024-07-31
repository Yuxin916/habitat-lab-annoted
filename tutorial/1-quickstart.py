"""Set up a PointNav task in which the agent is tasked
to go from a source location to a target location
到点导航任务
"""

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2

# keyboard actions
FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def example():
    env = habitat.Env(
        config=habitat.get_config("configs/tasks/pointnav_rgbd.yaml")  # FIXED
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations[0]["pointgoal_with_gps_compass"][0],  # 距离目标点的距离
        observations[0]["pointgoal_with_gps_compass"][1])) # 与目标点的角度
    # eg:
    # Destination, distance: 0.510518, theta(radians): -0.00
    # Destination, distance: 0.260522, theta(radians): -0.01
    # Destination, distance: 0.010737, theta(radians): -0.21

    # Setup the window to display with specific size
    # 创建窗口可视化
    window_name = "RGB"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 1200)  # Set the window size
    cv2.imshow("RGB", transform_rgb_bgr(observations[0]["rgb"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.MOVE_FORWARD
            ("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step([action])
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations[0]["pointgoal_with_gps_compass"][0],
            observations[0]["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations[0]["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.STOP
        and observations[0]["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()

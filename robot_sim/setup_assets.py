"""Regenerate deterministic simulator assets after a fresh checkout."""
from robot_sim.scripts.configure_robot import configure
from robot_sim.scripts.build_scenes import build
from robot_sim.rover import build as build_rover
if __name__ == "__main__":
    configure()
    build()
    build_rover()

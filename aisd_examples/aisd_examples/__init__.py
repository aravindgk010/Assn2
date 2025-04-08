from gymnasium.envs.registration import register

register(
    id="aisd_examples/RedBall-V0",
    entry_point="aisd_examples.envs:RedBallEnv",
)

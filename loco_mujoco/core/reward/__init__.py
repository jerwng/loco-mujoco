from .base import Reward
from .default import NoReward, TargetVelocityGoalReward, TargetXVelocityReward, LocomotionReward, VelocityWithHeadingGoalReward, LocomotionWithHeadingReward
from .trajectory_based import TargetVelocityTrajReward, MimicReward
from .utils import *

# register all rewards
NoReward.register()
TargetVelocityGoalReward.register()
TargetXVelocityReward.register()
TargetVelocityTrajReward.register()
MimicReward.register()
LocomotionReward.register()
VelocityWithHeadingGoalReward.register()
LocomotionWithHeadingReward.register()

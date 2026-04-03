import mujoco
from mujoco import MjSpec
from .unitreeGo2 import UnitreeGo2


class MjxUnitreeGo2(UnitreeGo2):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=8, ls_iterations=12,
                                     cone=mujoco.mjtCone.mjCONE_PYRAMIDAL,
                                     impratio=1,
                                     disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
               two capsules.
            2. Disable all contacts except: (a) feet and the floor, (b) the robot front geoms and named
               room object geoms.

        Args:
            spec (MjSpec): Mujoco specification.

        Returns:
            Modified Mujoco specification.

        """

        # --- 1. Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- 2. Define specific contact pairs ---
        spec.add_pair(geomname1="floor", geomname2="RL_foot")
        spec.add_pair(geomname1="floor", geomname2="RR_foot")
        spec.add_pair(geomname1="floor", geomname2="FL_foot")
        spec.add_pair(geomname1="floor", geomname2="FR_foot")

        # --- 3. Add front-to-room-object contact pairs ---
        # Exclude robot geoms (legs, front geoms themselves used as geomname1) and the floor.
        # Keep only the front sphere here because MJX does not implement cylinder-box contacts.
        robot_geom_names = {"front_cylinder", "front_sphere",
                    "FL_foot", "FR_foot", "RL_foot", "RR_foot"}
        skip_geom_names = {"floor"}
        front_geom_names = ["front_sphere"]

        for g in spec.geoms:
            if g.name and g.name not in robot_geom_names and g.name not in skip_geom_names:
                for front_name in front_geom_names:
                    spec.add_pair(geomname1=front_name, geomname2=g.name)

        return spec
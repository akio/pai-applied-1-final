"""ビューアのマウス操作 (ドラッグで回転、ホイールでズーム) の中心をロボットに合わせる。

Genesis のビューアは、マウス操作の中心をビューアを作ったときの注視点 (ViewerOptions.camera_lookat) に固定する。
viewer.set_camera_pose や follow_entity でカメラを動かしても中心は変わらないので、不整地の環境では
ワールド原点 (地形の隅) の遠い点を中心に回ってしまう。ここでは毎フレーム、中心をロボットの位置に置き直す。
中心はビューアの内部 (pyrender の Trackball) の値を直接書き換える (Genesis に設定する方法がないため)。
"""

import numpy as np

import genesis.utils.geom as gu

WORLD_UP = np.array([0.0, 0.0, 1.0])


class RobotOrbitCamera:
    """マウス操作の中心をロボットに置き、follow=True ならロボットが動いた分だけカメラも平行移動する。"""

    def __init__(self, viewer, smoothing=0.8):
        self.viewer = viewer
        self.smoothing = smoothing
        self.center = None

    def reset(self):
        """次の update で中心を今のロボット位置に置き直し、カメラをロボットに向ける (視点の切り替え・リセットの後に呼ぶ)。"""
        self.center = None

    def update(self, robot_pos, follow):
        robot_pos = np.asarray(robot_pos, dtype=np.float64)
        trackball = self.viewer._pyrender_viewer._trackball
        if self.center is None:
            center, delta = robot_pos, np.zeros(3)
            # カメラの位置はそのままで、ロボットを向かせる (マウスの回転は中心を向いたまま回る)
            eye = trackball.pose[:3, 3].copy()
            self.viewer.set_camera_pose(pose=gu.pos_lookat_up_to_T(eye, center, WORLD_UP))
            trackball._pose = trackball._n_pose.copy()
        else:
            center = self.smoothing * self.center + (1.0 - self.smoothing) * robot_pos
            delta = center - self.center
        self.center = center
        # ビューアの操作中の姿勢 (_pose: ドラッグ開始時, _n_pose: 表示中) と中心をそろって動かす
        if follow:
            for name in ("_pose", "_n_pose"):
                pose = getattr(trackball, name).copy()
                pose[:3, 3] += delta
                setattr(trackball, name, pose)
        trackball._target = center.copy()
        trackball._n_target = center.copy()

# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from numpy import typing as npt

Vector3 = npt.NDArray[np.float64]  # sized 3 array of floats


class AppState:
    def __init__(self):
        self.WIN_NAME = "RealSense"

        self.pitch = math.radians(-10)
        self.yaw = math.radians(-15)
        self.translation: Vector3 = np.array([0, 0, -1], dtype=np.float32)

        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.window_scale = 2
        self.color = True
        self.width = 0
        self.height = 0

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # Get stream profile and camera intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height
        self.width = w
        self.height = h

        # Processing blocks
        self.pc = rs.pointcloud()
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2**self.decimate)
        self.colorizer = rs.colorizer()

        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.WIN_NAME, w, h)
        cv2.setMouseCallback(self.WIN_NAME, mouse_cb, (self,))

    def reset(self):
        self.pitch = 0.0
        self.yaw = 0.0
        self.distance = 2.0
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues(np.array([self.pitch, 0, 0]))
        Ry, _ = cv2.Rodrigues(np.array([0, self.yaw, 0]))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

    def view(self, v: Vector3):
        """apply view transformation on vector array"""
        return np.dot(v - self.pivot, self.rotation) + self.pivot - self.translation

    def grid(
        self,
        out,
        pos: Vector3,
        rotation=np.eye(3),
        size=1,
        n=10,
        color=(0x80, 0x80, 0x80),
    ):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n + 1):
            x = -s2 + i * s
            self.line3d(
                out,
                self.view(pos + np.dot((x, 0, -s2), rotation)),
                self.view(pos + np.dot((x, 0, s2), rotation)),
                color,
            )
        for i in range(0, n + 1):
            z = -s2 + i * s
            self.line3d(
                out,
                self.view(pos + np.dot((-s2, 0, z), rotation)),
                self.view(pos + np.dot((s2, 0, z), rotation)),
                color,
            )

    def get_point(self, out, orig, intrinsics, d, color, x, y):
        p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
        self.line3d(out, orig, self.view(p), color)
        return p

    def frustum(self, out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view(np.array([0, 0, 0]))
        w, h = intrinsics.width, intrinsics.height

        for d in range(1, 6, 2):

            top_left = self.get_point(out, orig, intrinsics, d, color, 0, 0)
            top_right = self.get_point(out, orig, intrinsics, d, color, w, 0)
            bottom_right = self.get_point(out, orig, intrinsics, d, color, w, h)
            bottom_left = self.get_point(out, orig, intrinsics, d, color, 0, h)

            self.line3d(out, self.view(top_left), self.view(top_right), color)
            self.line3d(out, self.view(top_right), self.view(bottom_right), color)
            self.line3d(out, self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(out, self.view(bottom_left), self.view(top_left), color)

    @staticmethod
    def project(image, v):
        """project 3d vector array to 2d"""
        h, w = image.shape[:2]
        view_aspect = float(h) / w

        # ignore divide by zero for invalid depth
        with np.errstate(divide="ignore", invalid="ignore"):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * (w * view_aspect, h) + (
                w / 2.0,
                h / 2.0,
            )

        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj

    def line3d(self, out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        p0 = self.project(out, pt1.reshape(-1, 3))[0]
        p1 = self.project(out, pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        h, w = out.shape[0:2]
        rect = (0, 0, w, h)
        inside, p0, p1 = cv2.clipLine(rect, p0, p1)
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)

    def axes(self, out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(
            out, pos, pos + np.dot((0, 0, size), rotation), (0xFF, 0, 0), thickness
        )
        self.line3d(
            out, pos, pos + np.dot((0, size, 0), rotation), (0, 0xFF, 0), thickness
        )
        self.line3d(
            out, pos, pos + np.dot((size, 0, 0), rotation), (0, 0, 0xFF), thickness
        )

    def fill_pointcloud(self, out, verts, texcoords, color, decimate, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(out, v[s])
        else:
            proj = self.project(out, self.view(verts))

        if decimate != 1:
            proj *= 0.5**decimate

        h, w = out.shape[:2]

        np.nan_to_num(proj, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm

        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch - 1, out=u)
        np.clip(v, 0, cw - 1, out=v)

        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]

    def run(self) -> None:
        img_w = self.width
        img_h = self.height
        debug_w = img_w * self.window_scale
        debug_h = img_h * self.window_scale
        out = np.empty((img_h, img_w, 3), dtype=np.uint8)
        while True:
            # Grab camera data
            if not self.paused:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                depth_frame = self.decimate_filter.process(depth_frame)

                # Grab new intrinsics (may be changed by decimation)
                depth_intrinsics = rs.video_stream_profile(
                    depth_frame.profile
                ).get_intrinsics()

                # depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_colormap = np.asanyarray(
                    self.colorizer.colorize(depth_frame).get_data()
                )

                if self.color:
                    mapped_frame, color_source = color_frame, color_image
                else:
                    mapped_frame, color_source = depth_frame, depth_colormap

                points = self.pc.calculate(depth_frame)
                self.pc.map_to(mapped_frame)

                # Pointcloud data to arrays
                v, t = points.get_vertices(), points.get_texture_coordinates()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

            # Render
            now = time.time()

            out.fill(0)

            scale = self.decimate != 1 or self.window_scale != 1
            if not scale or (debug_h, debug_w) == (img_h, img_w):
                self.fill_pointcloud(out, verts, texcoords, color_source, self.decimate)
                show_img = out
            else:
                tmp = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                self.fill_pointcloud(tmp, verts, texcoords, color_source, self.decimate)
                tmp = cv2.resize(
                    tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST
                )
                np.putmask(out, tmp > 0, tmp)
                show_img = cv2.resize(
                    out, (debug_w, debug_h), interpolation=cv2.INTER_NEAREST
                )

            if any(self.mouse_btns):
                self.axes(show_img, self.view(self.pivot), self.rotation, thickness=4)

            self.grid(show_img, np.array([0, 0.5, 1]), size=1, n=10)
            self.frustum(show_img, depth_intrinsics)
            self.axes(
                show_img,
                self.view(np.array([0, 0, 0])),
                self.rotation,
                size=0.1,
                thickness=1,
            )

            dt = time.time() - now

            cv2.setWindowTitle(
                self.WIN_NAME,
                "RealSense (%dx%d) %dFPS (%.2fms) %s"
                % (img_w, img_h, 1.0 / dt, dt * 1000, "PAUSED" if self.paused else ""),
            )

            cv2.imshow(self.WIN_NAME, show_img)
            key = cv2.waitKey(1)

            if key == ord("r"):
                self.reset()

            if key == ord("p"):
                self.paused ^= True

            if key == ord("c"):
                self.color ^= True

            if key == ord("s"):
                cv2.imwrite("./out.png", out)

            if key == ord("e"):
                points.export_to_ply("./out.ply", mapped_frame)

            if (
                key in (27, ord("q"))
                or cv2.getWindowProperty(self.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0
            ):
                break

        # Stop streaming
        self.pipeline.stop()


def mouse_cb(event, x, y, flags, param):
    self: AppState = param[0]

    if event == cv2.EVENT_LBUTTONDOWN:
        self.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        self.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        self.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        self.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        self.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        self.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:
        h = self.height
        w = self.width
        dx = x - self.prev_mouse[0]
        dy = y - self.prev_mouse[1]

        if self.mouse_btns[0]:
            self.yaw += float(dx) / w * 2
            self.pitch -= float(dy) / h * 2

        elif self.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            self.translation -= np.dot(self.rotation, dp)

        elif self.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            self.translation[2] += dz
            self.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        self.translation[2] += dz
        self.distance -= dz

    self.prev_mouse = (x, y)


def main() -> None:
    AppState().run()


if __name__ == "__main__":
    main()

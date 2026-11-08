#!/usr/bin/env bash
# The system libraries DexVantage needs on an Ubuntu 22.04 or 24.04 runner without a GPU or a display
# (.github/workflows/*.yml: the tests run on 24.04, the Linux bundles build on 22.04).
#
#   OpenGL and EGL through Mesa: libEGL/libGL/libOpenGL (glvnd), Mesa's EGL vendor and its DRI drivers, which
#   include llvmpipe, the software renderer. MuJoCo's render worker (MUJOCO_GL=egl, dexvantage/validation/mjrender.py)
#   and PyOpenGL's EGL contexts (the occlusion cube maps, the self-test's OpenGL line) then render without a GPU.
#   Qt 6 (PyQt6 wheels): the libraries libQt6Gui and the platform plugins link against (read with ldd from the
#   PyQt6-Qt6 6.12.0 wheel), xkbcommon, fontconfig, freetype, D-Bus, GLib, and the X11/xcb libraries of the xcb
#   plugin (the tests use the offscreen plugin; PyInstaller bundles carry the xcb plugin and its libraries).
set -euo pipefail
sudo apt-get update -q
# GLib's package was renamed for the 64-bit time_t transition in Ubuntu 24.04; 22.04 has the old name
glib=libglib2.0-0t64
apt-cache show "$glib" > /dev/null 2>&1 || glib=libglib2.0-0
sudo apt-get install -y -q --no-install-recommends \
  libegl1 libgl1 libglx0 libopengl0 libegl-mesa0 libgl1-mesa-dri libgbm1 \
  libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libfreetype6 libdbus-1-3 "$glib" \
  libx11-6 libx11-xcb1 libxcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 libxcb-util1 libxcb-xfixes0 \
  libxcb-xinerama0 libxcb-xkb1

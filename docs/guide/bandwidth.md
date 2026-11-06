# Bandwidth, hosts and power: a worked example

Can the host carry every stream, does each USB camera get the current it needs, and how much does a campaign
record? This walk-through uses the UR5e + Robotiq 2F-85 sample ("Wrist D405" on the flange, "Overhead D435" on the
cage). The model, its sources and its checks are in `docs/models/bandwidth.md`. The figures below are what this build
computes on the sample.

## 1. Open the sample and run the study

    dexvantage --sample ur5e

Analysis ▸ Bandwidth, hosts and power opens the page; the study runs live after every edit. In batch mode:

    dexvantage --sample ur5e --bandwidth bandwidth.json

The sample has no host yet, so the headline reads **—, no hosts**. The data rates are already there. Neither sensor's
link names streams, so each sends its default streams: the depth stream and the RGB stream at their largest size and
the listed rate nearest 30 fps (a default):

| Sensor | Streams | Payload | On the wire (+ 5 %, a default) |
|---|---|---|---|
| Wrist D405 | depth 1280 × 720 @ 30 Z16, RGB 1280 × 720 @ 30 YUY2 | 884.7 Mbit/s | 929.0 Mbit/s |
| Overhead D435 | depth 1280 × 720 @ 30 Z16, RGB 1920 × 1080 @ 30 YUY2 | 1437.7 Mbit/s | 1509.6 Mbit/s |

For example 1920 × 1080 × 30 fps × 16 bits per pixel = 995.3 Mbit/s for the D435's colour stream: every figure is a
stream mode of the Intel datasheet the library entry cites.

## 2. Give the rig a host

**Apply** makes the page's suggestion, one undo step: it adds "Host (assumed)" with one "USB 3 controller (assumed)"
and plugs both cameras into it. The headline now reads **68 %, within budget**:

- the controller carries 2439 Mbit/s of 3600 Mbit/s, 1161 Mbit/s of margin. 3600 Mbit/s is the USB 3.2
  specification's effective bandwidth for Gen 1 ("450 MB/sec or less", §4.4.11), an upper bound: the controller has
  no library entry, so Gen 1, the lowest generation, is assumed and the topology tree says so;
- each camera's own link: the D405 at 26 %, the D435 at 42 % of the same Gen 1 figure (both are "USB 3.1 Gen 1"
  devices);
- bus power: the D435 states 3.40 W, 681 mA at 5 V, of the 900 mA a USB 3 port gives; the D405 296 mA.

To use your machine instead, pick its entry in the host list (for example "USB 3.2 Gen 2 host port") and **Add host**:
a Gen 2 controller is compared with 8800 Mbit/s, and each camera's link stays at Gen 1, the camera's own generation.
If you have measured a controller (a bulk-transfer test, `usbtop` while streaming), select it in the tree, enter the
figure and its source under **Practical bandwidth** and **Set**: that figure then replaces the specification's bound.

## 3. A bus-powered hub

Select the controller, untick **Powered** and **Add hub**; then plug both cameras into "Hub 1" in the *Plugged into*
column (each change is one undo step). The page now reads **over budget**:

> Bus-powered Hub 1: the devices on it state 977 mA, more than the 900 mA its upstream port gives (its own
> electronics not counted). Use a powered hub or move a device to another port.

A bus-powered USB 3 hub shares its own port's 900 mA among its ports (USB 3.2 §10.14). Tick **Powered** on a new hub
(or remove this one): a powered hub gives every port the full 900 mA and the finding goes. A hub that shares its
controller with other devices is flagged too ("on a shared controller"), because everything on it shares that
controller's bandwidth.

## 4. A USB 2 port

Plugged into a USB 2.0 controller instead, the same two cameras need 2439 Mbit/s of 426 Mbit/s (USB 2.0's bulk limit,
53 248 000 bytes/s, Table 5-10): 572 %, **over budget**, with each camera's link over its limit too ("depth + RGB send
1510 Mbit/s over 426 Mbit/s"). Choosing 640 × 480 for both on the Sensors page brings each to 310 Mbit/s on the wire,
73 % of its own link, but the two together still need 145 % of the one controller: move one camera to another
controller, or record fewer streams (the link's stream list).

## 5. The recording budget

At the default 60 s episodes and 100 episodes per campaign, the two cameras record 299.0 MB/s (raw as streamed, plus
3 % container overhead, both defaults): **17.94 GB per episode, 1.79 TB per campaign**. The recorder's sustained
write speed is "not stated" until you enter it: type your disk's sustained figure (after its write cache fills) and
its source under **Recording** and **Set**. With 640 × 480 streams, 200 episodes and a 500 MB/s, 2000 GB disk, the
study reads 75.9 MB/s recorded, 98.7 MB/s sustained write needed (× 1.3 headroom, a default), 4.56 GB per episode
and 438 episodes per disk.

## 6. What the study cannot tell you

Sensors whose maker states no bits per pixel (MJPEG streams, the Luxonis cameras, the DIGIT and GelSight Mini tactile
sensors) have no data rate until you enter a measured one; the page marks them and the controller's load is then a
lower bound. GMSL deserialisers have no stated link rate in the library, so a GMSL camera's link is not checked until
you enter the deserialiser's practical figure. The full list is in the model's "Known limits".

## Export

**Export…** writes the result as JSON with its provenance block: every sensor's streams and rates, every controller's
load, capacity, source and margin, the hubs, the disk budget, the bus power and the findings.

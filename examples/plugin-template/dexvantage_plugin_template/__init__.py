"""DexVantage plugin template: one study (aim at the task volume), its metrics and its page.

DexVantage finds this package through its entry point (pyproject.toml):

    [project.entry-points."dexvantage.plugins"]
    aim_template = "dexvantage_plugin_template:register"

and calls register(api) once at start-up. Everything is registered through api, so DexVantage knows what this plugin
added (Help ▸ Plugins) and can take it away again if loading fails. Copy this folder, rename the package and the ids,
and replace the study with your own.
"""
from __future__ import annotations

__version__ = '0.1.0'
REQUIRES_API = 1                 # the plugin interface version this was written for (dexvantage.plugin_sdk.API_VERSION)


def register(api):
    from . import aim

    api.study(aim.STUDY_ID, aim.TITLE, quick=True, batch='aim-check')(aim.aim)

    @api.metric('aim.worst_deg', 'Worst camera aim off the task volume', '°', 'lower', study=aim.STUDY_ID,
                note='the largest angle between a camera\'s viewing direction and its direction to the volume centre',
                tags=('headline',))
    def _worst(doc):
        return doc['summary']['worst_deg']

    @api.metric('aim.share_aimed', 'Cameras aimed at the task volume', '%', 'higher', study=aim.STUDY_ID,
                note='share of the enabled cameras within max_angle_deg of the volume centre (template default 20°)')
    def _share(doc):
        return doc['summary']['share_aimed_pct']

    # the page is named as "module:Class": DexVantage imports it only when the window is built (no Qt headless)
    api.page('aim_check', aim.TITLE, 'dexvantage_plugin_template.page:AimPage', menu='Analysis', icon='target',
             workspace='placement', order=210)

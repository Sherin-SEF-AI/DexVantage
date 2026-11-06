"""The shared dataset shape (core/datasets/episode.py): registering a reader, sniffing, plain errors."""
import numpy as np
import pytest

from dexvantage.core import datasets as DS


def test_register_open_and_errors(tmp_path):
    class Tiny(DS.Dataset):
        format, version = 'tiny-test', 'v1'

        def __init__(self, path):
            self.path, self.info = path, {}

        def episodes(self):
            return [DS.EpisodeRef(0, 'e0', 3)]

        def load(self, index):
            cam = DS.CameraStream('cam', 4, 2, 10.0, np.array([0.0, 0.1, 0.2]))
            return DS.Episode('e0', 0, np.array([0.0, 0.1, 0.2]), np.zeros((3, 2)), ['a', 'b'], cameras={'cam': cam})

    f = tmp_path / 'x.tiny'
    f.write_text('tiny')
    DS.register_reader('tiny-test', lambda p: str(p).endswith('.tiny'))(Tiny)
    try:
        ds = DS.open_dataset(str(f))
        assert ds.format == 'tiny-test' and ds.episodes()[0].length == 3
        ep = ds.load(0)
        assert ep.duration_s == pytest.approx(0.2) and ep.cameras['cam'].n_frames == 3
        with pytest.raises(DS.DatasetError, match='no video decoder'):
            ep.cameras['cam'].frames()
        other = tmp_path / 'y.unknown'
        other.write_text('?')
        with pytest.raises(DS.DatasetError, match='not a dataset format'):
            DS.open_dataset(str(other))
        with pytest.raises(DS.DatasetError, match='no dataset'):
            DS.open_dataset(str(tmp_path / 'missing'))
    finally:
        DS.READERS[:] = [r for r in DS.READERS if r[0] != 'tiny-test']
